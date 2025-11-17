from Gaussian_real import generate_gaussian_experiment_real

if __name__ == '__main__':

    # Base output directory for the real-noise experiment results
    root_dir_output_real = f'/workspace/data/output/real_noisy/test'

    # Directory containing the general input images used in the experiment
    dir_noisy_real_images = f'/workspace/data/input/noisy_real_images'
    dir_non_noisy_real_images = f'/workspace/data/input/non_noisy_real_images'
  
      
    # Dictionary of parameters passed to the experiment generator
    parameters = {

        # Paths for reading input and saving results
        'root_dir_output_real': root_dir_output_real,
        'dir_noisy_real_images': dir_noisy_real_images,
        'dir_non_noisy_real_images': dir_non_noisy_real_images,
        
        # Output folders for each filtering method
        'dir_out_nlm': f'{root_dir_output_real}/NLM',
        'dir_out_geonlm': f'{root_dir_output_real}/GEONLM',
        'dir_out_bm3d': f'{root_dir_output_real}/BM3D',
        'dir_out_results': f'{root_dir_output_real}/results',

        # Filenames for serialized results (pickle/XLSX)
        'name_pickle_nlm_output_real': 'array_nln_real_filtereds.pkl',
        'name_pickle_results_gnlm_bm3d_output_real': 'results_gaussian_gnlm_bm3d_real.pkl',
        'name_results_xlsx_nlm_gnlm_bm3d_output_real': 'gnlm_bm3d_real_filtereds.xlsx',

        # Algorithmic parameters used internally by the experiment
        'f': 4,      # Patch radius
        't': 7,      # Search window radius
        'alpha': 0.5, # Geometric weight (for GEO-NLM)
        'nn': 10,     # Number of nearest neighbors / similar patches
    }

    # Execute the real-noise Gaussian experiment
    generate_gaussian_experiment_real(parameters)
