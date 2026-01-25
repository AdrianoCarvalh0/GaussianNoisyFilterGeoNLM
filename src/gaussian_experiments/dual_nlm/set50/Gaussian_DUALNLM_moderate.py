import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skimage.restoration import estimate_sigma
import numpy as np
import cupy as cp
import skimage
from functions.Utils import (read_directories, save_pickle, save_results_to_xlsx, load_pickle, is_low_noise_or, get_multiplier)
from functions.noisy_functions import (add_low_noise_gaussian, add_moderate_noise_gaussian, add_high_noise_gaussian)
from functions.nlm_functions import (compute_adaptive_q, select_best_h_using_adaptive_q)
from functions.geonlm_functions import run_geonlm_pipeline
import time
from bm3d import bm3d, BM3DProfile
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def divergenciaKL(P1m, P1v, P2m, P2v):
    dKL = np.sum( (1/(4*P1v*P2v))*( (P1v - P2v)**2 + ((P1m - P2m)**2)*(P1v + P2v) ) )
    return dKL


def Bhattacharyya(P1m, P1v, P2m, P2v):
    cBhat = np.sqrt( (2*np.sqrt(P1v)*np.sqrt(P2v)/(P1v + P2v) )*np.exp( -0.25* ( ((P1m - P2m)**2)/(P1v + P2v) ) ) )
    dBhat = np.sum( -np.log(cBhat) )
    return dBhat


def Hellinger(P1m, P1v, P2m, P2v):
    cBhat = np.sqrt( (2*np.sqrt(P1v)*np.sqrt(P2v)/(P1v + P2v) )*np.exp( -0.25* ( ((P1m - P2m)**2)/(P1v + P2v) ) ) )
    dHell = np.sum( 1 - cBhat )
    return dHell


def NLM(ruidosa, h, f, t):
    m, n = ruidosa.shape
    filtrada = np.zeros((m, n))
    img_n = np.pad(ruidosa, ((f, f), (f, f)), 'symmetric')

    for i in range(m):
        for j in range(n):
            im = i + f
            jn = j + f
            W1 = img_n[im-f:(im+f)+1, jn-f:(jn+f)+1]

            rmin = max(im-t, f)
            rmax = min(im+t, m+f)
            smin = max(jn-t, f)
            smax = min(jn+t, n+f)

            NL = 0
            Z = 0

            for r in range(rmin, rmax):
                for s in range(smin, smax):
                    W2 = img_n[r-f:(r+f)+1, s-f:(s+f)+1]
                    d2 = np.sum((W1 - W2)*(W1 - W2))
                    sij = np.exp(-d2/(h**2))
                    Z = Z + sij
                    NL = NL + sij*img_n[r, s]

            filtrada[i, j] = NL/Z

    return filtrada


def NLM_KL(noised, h, f, t):
    m, n = noised.shape
    filtrada = np.zeros((m, n))
    matriz_m = np.zeros((m, n))
    matriz_v = np.zeros((m, n))

    matriz_m = NLM(noised, 70, 4, 7)
    img_n = np.pad(noised, ((f, f), (f, f)), 'symmetric')

    for i in range(m):
        for j in range(n):
            im = i + f
            jn = j + f
            matriz_v[i, j] = img_n[im-f:im+f+1, jn-f:jn+f+1].var()

    matriz_m = np.pad(matriz_m, ((f, f), (f, f)), 'symmetric')
    matriz_v = np.pad(matriz_v, ((f, f), (f, f)), 'symmetric')

    for i in range(m):
        for j in range(n):
            im = i + f
            jn = j + f
            W1m = matriz_m[im-f:(im+f)+1, jn-f:(jn+f)+1]
            W1v = matriz_v[im-f:(im+f)+1, jn-f:(jn+f)+1]

            rmin = max(im-t, f)
            rmax = min(im+t, m+f)
            smin = max(jn-t, f)
            smax = min(jn+t, n+f)

            NL = 0
            Z = 0

            for r in range(rmin, rmax):
                for s in range(smin, smax):
                    W2m = matriz_m[r-f:(r+f)+1, s-f:(s+f)+1]
                    W2v = matriz_v[r-f:(r+f)+1, s-f:(s+f)+1]

                    dKL = divergenciaKL(W1m, W1v, W2m, W2v)
                    sij = np.exp(-dKL/(h**2))
                    Z = Z + sij
                    NL = NL + sij*img_n[r, s]

            filtrada[i, j] = NL/Z

    return filtrada


def generate_gaussian_experiment_moderate_dual_nlm(parameters):
    """
    Run the moderate-noise Gaussian denoising experiment using NLM, GEO-NLM, and BM3D.
    """
    root_dir_output_moderate = parameters['root_dir_output_moderate']
    dir_images_general = parameters['dir_images_general']
    dir_out_results = parameters['dir_out_results']
    dir_out_dualnlm = parameters['dir_out_dualnlm']
    name_pickle_dual_nlm_output_moderate = parameters['name_pickle_dual_nlm_output_moderate']
    name_results_xlsx_dual_nlm_output_moderate = parameters['name_results_xlsx_dual_nlm_output_moderate']
    pickle_results_summary_moderate = parameters['pickle_results_summary_moderate']
    pickle_results_cameraman = parameters['pickle_results_cameraman']
    f = parameters['f']
    t = parameters['t']
    h = parameters['h']
    alpha = parameters['alpha']

    array_dir = read_directories(dir_images_general)
    array_dual_nln_moderate_filtereds = []

    vector = load_pickle('array_pickle_nlm', pickle_results_summary_moderate)
    cameraman = load_pickle('pickle_cameraman', pickle_results_cameraman)    

    for vect in vector:
        file_name = vect['file_name']        
        if file_name == '0.gif':
            img_noisse_gaussian_np = cameraman[0]['img_noisse_gaussian_np']
        else:
            img_noisse_gaussian_np = vect['img_noisse_gaussian_np']        
        img = skimage.io.imread(f'{dir_images_general}/{file_name}')

        if img.ndim == 4:
            img = img[0]

        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]

        if img.ndim == 3 and img.shape[-1] == 3:
            img = skimage.color.rgb2gray(img)

        if img.dtype.kind == 'f':
            img = (np.clip(img, 0, 1) * 255).astype(np.float32)
        else:
            img = img.astype(np.float32)

        filtrada = NLM_KL(img_noisse_gaussian_np, h, f, t)
        result_uint8 = np.clip(filtrada, 0, 255).astype(np.uint8)

        skimage.io.imsave(
            f'{dir_out_dualnlm}/{file_name}',
            np.clip(filtrada, 0, 255).astype(np.uint8)
        )

        psnr = peak_signal_noise_ratio(img, result_uint8, data_range=255)
        ssim = structural_similarity(img, result_uint8, data_range=255)
        score = alpha * psnr + (1 - alpha) * (ssim * 100)

        print(f"PSNR = {psnr:.2f} | SSIM = {ssim:.4f} | Score = {score:.2f}")

        dct = {
            'filtrada': filtrada,
            'psnr': psnr,
            'ssim': ssim,
            'score': score,
            'file_name': file_name,
        }
        array_dual_nln_moderate_filtereds.append(dct)

    save_pickle(array_dual_nln_moderate_filtereds, dir_out_results, name_pickle_dual_nlm_output_moderate)
    save_results_to_xlsx(
        dir_out_results,
        name_results_xlsx_dual_nlm_output_moderate
    )