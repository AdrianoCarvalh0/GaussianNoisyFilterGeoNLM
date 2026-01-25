

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

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



'''
Parâmetros:

	P1m, P1v: patches centrais de referência das médias e variâncias
	P2m, P2v: patches das médias e variâncias pertencente a janela de busca
'''
def divergenciaKL(P1m, P1v, P2m, P2v):

	# Versão mais rápida de calcular
	dKL = np.sum( (1/(4*P1v*P2v))*( (P1v - P2v)**2 + ((P1m - P2m)**2)*(P1v + P2v) ) )

	return dKL

'''
Parâmetros:

	P1m, P1v: patches centrais de referência das médias e variâncias
	P2m, P2v: patches das médias e variâncias pertencente a janela de busca
'''
def Bhattacharyya(P1m, P1v, P2m, P2v):

	# Calcula o coeficiente de Bhattacharyya (np.sum é aqui ou embaixo?)
	cBhat = np.sqrt( (2*np.sqrt(P1v)*np.sqrt(P2v)/(P1v + P2v) )*np.exp( -0.25* ( ((P1m - P2m)**2)/(P1v + P2v) ) ) )
	# Calcula distância de Bhattacharyya
	dBhat = np.sum( -np.log(cBhat) )

	return dBhat

'''
Parâmetros:

	P1m, P1v: patches centrais de referência das médias e variâncias
	P2m, P2v: patches das médias e variâncias pertencente a janela de busca
'''
def Hellinger(P1m, P1v, P2m, P2v):

	# Calcula o coeficiente de Bhattacharyya (np.sum é aqui ou embaixo?)
	cBhat = np.sqrt( (2*np.sqrt(P1v)*np.sqrt(P2v)/(P1v + P2v) )*np.exp( -0.25* ( ((P1m - P2m)**2)/(P1v + P2v) ) ) )
	# Calcula distância de Hellinger
	dHell = np.sum( 1 - cBhat )

	return dHell


'''
Parâmetros:

	img: imagem ruidosa de entrada
	h: parâmetro que controla o grau de suavização (quanto maior, mais suaviza)
	f: tamanho do patch (2f + 1 x 2f + 1) -> se f = 3, então patch é 7 x 7
	t: tamanho da janela de busca (2t + 1 x 2t + 1) -> se t = 10, então janela de busca é 21 x 21

'''
def NLM(img, h, f, t):

	# Dimenssões espaciais da imagem
	m, n = img.shape

	# Cria imagem de saída
	filtrada = np.zeros((m, n))

	# Problema de valor de contorno: replicar bordas
	img_n = np.pad(img, ((f, f), (f, f)), 'symmetric')

	# Loop principal do NLM
	for i in range(m):
		for j in range(n):

			im = i + f;   # compensar a borda adicionada artificialmente
			jn = j + f;   # compensar a borda adicionada artificialmente

        	# Obtém o patch ao redor do pixel corrente
			W1 = img_n[im-f:(im+f)+1, jn-f:(jn+f)+1]

        	# Calcula as bordas da janela de busca para o pixel corrente
			rmin = max(im-t, f);  # linha inicial
			rmax = min(im+t, m+f);  # linha final
			smin = max(jn-t, f);  # coluna inicial
			smax = min(jn+t, n+f);  # coluna final

        	# Calcula média ponderada
			NL = 0      # valor do pixel corrente filtrado
			Z = 0       # constante normalizadora

        	# Estima h localmente dentro de cada janela de busca (local) - não é usual
        	# h = np.sqrt(img_n[rmin:rmax, smin:smax].var())

        	# Loop para todos os pixels da janela de busca
			for r in range(rmin, rmax):
				for s in range(smin, smax):

                	# Obtém o patch ao redor do pixel a ser comparado
					W2 = img_n[r-f:(r+f)+1, s-f:(s+f)+1]

                	# Calcula o quadrado da distância Euclidiana
					d2 = np.sum((W1 - W2)*(W1 - W2))

                	# Calcula a medida de similaridade
					sij = np.exp(-d2/(h**2))

                	# Atualiza Z e NL
					Z = Z + sij
					NL = NL + sij*img_n[r, s]

        	# Normalização do pixel filtrado
			filtrada[i, j] = NL/Z

	return filtrada




'''
Parâmetros:

	img: imagem ruidosa de entrada
	h: parâmetro que controla o grau de suavização (quanto maior, mais suaviza)
	f: tamanho do patch (2f + 1 x 2f + 1) -> se f = 3, então patch é 7 x 7
	t: tamanho da janela de busca (2t + 1 x 2t + 1) -> se t = 10, então janela de busca é 21 x 21

'''
def NLM_KL(img, h, f, t):

	# Dimenssões espaciais da imagem
	m, n = img.shape

	# Cria imagem de saída
	filtrada = np.zeros((m, n))

	# Cria matrizes para armazenar médias e variâncias
	matriz_m = np.zeros((m, n))
	matriz_v = np.zeros((m, n))

	# Estima médias de maneira não-local
	# O parâmetro h dessa pré filtragem é diferente do h usado no filtro proposto! Em geral, bem maior!

	# Estima médias de maneira não local com NLM padrão
	matriz_m = NLM(img, 70, 4, 7)		# default = 70, testei 60 em alguns [50, 55, 60, 65, 70]

	# Problema de valor de contorno: replicar bordas
	img_n = np.pad(img, ((f, f), (f, f)), 'symmetric')

	# Estima variâncias locais
	for i in range(m):
		for j in range(n):
			im = i + f
			jn = j + f
			matriz_v[i, j] = img_n[im-f:im+f+1, jn-f:jn+f+1].var()

	# Replica bordas
	matriz_m = np.pad(matriz_m, ((f, f), (f, f)), 'symmetric')
	matriz_v = np.pad(matriz_v, ((f, f), (f, f)), 'symmetric')

	# Loop principal do NLM
	for i in range(m):
		for j in range(n):

			im = i + f;   # compensar a borda adicionada artificialmente
			jn = j + f;   # compensar a borda adicionada artificialmente

        	# Obtém o patch ao redor do pixel corrente em matriz_m e matriz_v
			W1m = matriz_m[im-f:(im+f)+1, jn-f:(jn+f)+1]
			W1v = matriz_v[im-f:(im+f)+1, jn-f:(jn+f)+1]

        	# Calcula as bordas da janela de busca para o pixel corrente
			rmin = max(im-t, f);  # linha inicial
			rmax = min(im+t, m+f);  # linha final
			smin = max(jn-t, f);  # coluna inicial
			smax = min(jn+t, n+f);  # coluna final

        	# Calcula média ponderada
			NL = 0      # valor do pixel corrente filtrado
			Z = 0       # constante normalizadora

        	# Loop para todos os pixels da janela de busca
			for r in range(rmin, rmax):
				for s in range(smin, smax):

                	# Obtém o patch ao redor do pixel a ser comparado em matriz_m e matriz_v
					W2m = matriz_m[r-f:(r+f)+1, s-f:(s+f)+1]
					W2v = matriz_v[r-f:(r+f)+1, s-f:(s+f)+1]

                	# Calcula a divergência KL simetrizada
					dKL = divergenciaKL(W1m, W1v, W2m, W2v)
					#dKL = Bhattacharyya(W1m, W1v, W2m, W2v)
					#dKL = Hellinger(W1m, W1v, W2m, W2v)

                	# Calcula a medida de similaridade
					sij = np.exp(-dKL/(h**2))

                	# Atualiza Z e NL
					Z = Z + sij
					NL = NL + sij*img_n[r, s]

        	# Normalização do pixel filtrado
			filtrada[i, j] = NL/Z

	return filtrada


def generate_gaussian_experiment_high_25_dual_nlm(parameters):
    """
    Run the high_25-noise Gaussian denoising experiment using NLM, GEO-NLM, and BM3D.
    
    The function:
      1. Reads all images in dir_images_general.
      2. Adds low-level Gaussian noise.
      3. Estimates sigma and computes an adaptive NLM parameter h.
      4. Runs NLM and stores intermediate results in a pickle file.
      5. Reloads those results to run GEO-NLM and BM3D.
      6. Computes PSNR, SSIM, and a custom score for each method.
      7. Saves filtered images and metrics (pickle + XLSX).
    """

    # Unpack configuration parameters
    root_dir_output_high_25 = parameters['root_dir_output_high_25']
    dir_images_general = parameters['dir_images_general']   
    dir_out_results = parameters['dir_out_results']
    dir_out_dualnlm = parameters['dir_out_dualnlm']
    name_pickle_dual_nlm_output_high_25 = parameters['name_pickle_dual_nlm_output_high_25']
    name_results_xlsx_dual_nlm_output_high_25 = parameters['name_results_xlsx_dual_nlm_output_high_25']
    pickle_results_summary_high_25 = parameters['pickle_results_summary_high_25']
    f = parameters['f']        # Patch radius (NLM)
    t = parameters['t']        # Search window radius (NLM ) 
    h = parameters['h']        # suavization parameter (NLM_KL)
    alpha = parameters['alpha']  # weight of mixed score
	
    # List all input image filenames in the general image directory
    array_dir = read_directories(dir_images_general)

    # Will store intermediate NLM results for all images
    array_dual_nln_high_25_filtereds = []

    vector = load_pickle('array_pickle_nlm', pickle_results_summary_high_25)
	    
    for vect in vector:
        file_name = vect['file_name']
        img_noisse_gaussian_np = vect['img_noisse_gaussian_np']
        estimated_sigma_gaussian_np = vect['estimated_sigma_gaussian']		

        # Read image from disk
        img = skimage.io.imread(f'{dir_images_general}/{file_name}')        

        # If the image has 4 dimensions (e.g. multi-page TIFF), use only the first slice
        if img.ndim == 4:
            img = img[0]

        # If the image has an alpha channel (RGBA), discard the alpha and keep RGB
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]

        # Convert RGB to grayscale if the image is color
        if img.ndim == 3 and img.shape[-1] == 3:
            img = skimage.color.rgb2gray(img)

        # Ensure the image is in [0, 255] as float32
        if img.dtype.kind == 'f':
            # If already float, clip to [0,1] then scale to [0,255]
            img = (np.clip(img, 0, 1) * 255).astype(np.float32)
        else:
            # If integer, just cast to float32
            img = img.astype(np.float32)

        m, n = img.shape  # Image dimensions (not used later, but kept for clarity)

        filtrada = NLM_KL(img_noisse_gaussian_np, h, f, t)      

        result_uint8 = np.clip(filtrada, 0, 255).astype(np.uint8)  

        # Save NLM-filtered image to disk
        skimage.io.imsave(
            f'{dir_out_dualnlm}/{file_name}',
            np.clip(filtrada, 0, 255).astype(np.uint8)
        )
		  # Quality metrics
        psnr = peak_signal_noise_ratio(img, result_uint8, data_range=255)
        ssim = structural_similarity(img, result_uint8, data_range=255)

        # Mixed score (PSNR + scaled SSIM)
        score = alpha * psnr + (1 - alpha) * (ssim * 100)
        print(f"PSNR = {psnr:.2f} | SSIM = {ssim:.4f} | Score = {score:.2f}")

        dct = {

            'filtrada': filtrada,   
            'psnr': psnr,
            'ssim': ssim,
            'score': score,
            'file_name': file_name,
        }
        array_dual_nln_high_25_filtereds.append(dct)

    # Save all NLM results (for all images) to a pickle file
    save_pickle(array_dual_nln_high_25_filtereds, dir_out_results, name_pickle_dual_nlm_output_high_25)

    # Export all results (NLM, GEO-NLM, BM3D) to an XLSX spreadsheet
    save_results_to_xlsx(       
        dir_out_results,
        name_results_xlsx_dual_nlm_output_high_25
    )
