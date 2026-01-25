from Gaussian_dual_nlm_real import generate_gaussian_dual_nlm_experiment_real

if __name__ == '__main__':

    # Base output directory for the real-noise experiment results
    root_dir_output_real = f'/workspace/data/output/pg_noisy'

    # Directory containing the general input images used in the experiment
    dir_noisy_real_images = f'/workspace/data/input/pg'
    dir_non_noisy_real_images = f'/workspace/data/input/pg_noisy'
  
      
    # Dictionary of parameters passed to the experiment generator
    parameters = {

        # Paths for reading input and saving results
        'root_dir_output_real': root_dir_output_real,
        'dir_noisy_real_images': dir_noisy_real_images,
        'dir_non_noisy_real_images': dir_non_noisy_real_images,
        
        # Output folders for each filtering method
        'dir_out_dual_nlm': f'{root_dir_output_real}/DUALNLM',
        'dir_out_results': f'{root_dir_output_real}/results',

        # Filenames for serialized results (pickle/XLSX)
        'name_pickle_dual_nlm_output_real': 'array_dual_nln_real_filtereds.pkl',
        'name_pickle_results_dual_nlm_output_real': 'results_gaussian_dual_nlm_real.pkl',
        'name_results_xlsx_dual_nlm_output_real': 'dual_nlm_real_filtereds.xlsx',

        # Algorithmic parameters used internally by the experiment
        'f': 4,      # Patch radius
        't': 7,      # Search window radius
        'h': 3.0,    # Filtering parameter
        'alpha': 0.5, # Geometric weight (for GEO-NLM)
        'nn': 10,     # Number of nearest neighbors / similar patches
    }

    # Execute the real-noise Gaussian experiment
    generate_gaussian_dual_nlm_experiment_real(parameters)
