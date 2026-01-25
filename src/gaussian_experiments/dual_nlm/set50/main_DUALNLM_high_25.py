import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Gaussian_DUALNLM_high_25 import generate_gaussian_experiment_high_25_dual_nlm

from functions.Utils import ensure_output_dirs

if __name__ == '__main__':

    # Base output directory for the high-noise experiment results
    root_dir_output_high_25 = Path('/workspace/data/output/set50/high_noisy_25') 

    # Directory containing the input images used in the experiment
    dir_images_general = Path('/workspace/data/input/general_images')

    # Ensure required output directories exist
    ensure_output_dirs(root_dir_output_high_25)

    # Dictionary of parameters passed to the experiment generator
    parameters = {

        # Paths for reading input and saving results
        'root_dir_output_high_25': str(root_dir_output_high_25),
        'dir_images_general': str(dir_images_general),

        # Output folders for each filtering method

        'dir_out_dualnlm': str(root_dir_output_high_25 / 'DUALNLM'),
        'dir_out_results': str(root_dir_output_high_25 / 'results'),

        # Filenames for serialized results (pickle/XLSX)
        'name_pickle_dual_nlm_output_high_25': 'array_dual_nlm_high_25_filtereds.pkl',    
        'name_results_xlsx_dual_nlm_output_high_25':'dual_nlm_high_25_filtereds.xlsx',
        'pickle_results_summary_high_25': '/workspace/data/output/set50/high_noisy_25/results/array_nln_high_25_filtereds.pkl',
        'pickle_results_cameraman': '/workspace/data/output/set50/high_noisy_25/results/array_nln_high_25_filtereds_cameraman.pkl',

        # Algorithmic parameters used internally by the experiment
        'f': 4,        # Patch radius
        't': 7,        # Search window radius    
        'h': 3.0,      # suavization parameter
        'alpha': 0.5,  # Geometric weight (for GEO-NLM)
    }

    # Execute the high_25-noise Gaussian experiment
    generate_gaussian_experiment_high_25_dual_nlm(parameters)
