import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from Gaussian_DUALNLM import generate_gaussian_experiment_low_dual_nlm

from functions.Utils import ensure_output_dirs

if __name__ == '__main__':

    # Base output directory for the low-noise experiment results
    root_dir_output_low = Path('/workspace/data/output/set12/low_noisy') 

    # Directory containing the input images used in the experiment
    dir_images_general = Path('/workspace/data/input/set12')

    # Ensure required output directories exist
    ensure_output_dirs(root_dir_output_low)

    # Dictionary of parameters passed to the experiment generator
    parameters = {

        # Paths for reading input and saving results
        'root_dir_output_low': str(root_dir_output_low),
        'dir_images_general': str(dir_images_general),

        # Output folders for each filtering method

        'dir_out_dualnlm': str(root_dir_output_low / 'DUALNLM'),
        'dir_out_results': str(root_dir_output_low / 'results'),

        # Filenames for serialized results (pickle/XLSX)
        'name_pickle_dual_nlm_output_low': 'array_dual_nlm_low_filtereds.pkl',    
        'name_results_xlsx_dual_nlm_output_low':'dual_nlm_low_filtereds.xlsx',
        'pickle_results_summary_low': '/workspace/data/output/set12/low_noisy/results/array_nln_low_filtereds.pkl',

        # Algorithmic parameters used internally by the experiment
        'f': 4,        # Patch radius
        't': 7,        # Search window radius    
        'h': 3.0,      # suavization parameter
        'alpha': 0.5,  # Geometric weight (for GEO-NLM)
    }

    # Execute the low-noise Gaussian experiment
    generate_gaussian_experiment_low_dual_nlm(parameters)
