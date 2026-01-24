import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from Gaussian_DUALNLM_high import generate_gaussian_experiment_high_dual_nlm

from functions.Utils import ensure_output_dirs

if __name__ == '__main__':

    # Base output directory for the high-noise experiment results
    root_dir_output_high = Path('/workspace/data/output/set12/high_noisy') 

    # Directory containing the input images used in the experiment
    dir_images_general = Path('/workspace/data/input/set12')

    # Ensure required output directories exist
    ensure_output_dirs(root_dir_output_high)

    # Dictionary of parameters passed to the experiment generator
    parameters = {

        # Paths for reading input and saving results
        'root_dir_output_high': str(root_dir_output_high),
        'dir_images_general': str(dir_images_general),

        # Output folders for each filtering method

        'dir_out_dualnlm': str(root_dir_output_high / 'DUALNLM'),
        'dir_out_results': str(root_dir_output_high / 'results'),

        # Filenames for serialized results (pickle/XLSX)
        'name_pickle_dual_nlm_output_high': 'array_dual_nlm_high_filtereds.pkl',    
        'name_results_xlsx_dual_nlm_output_high':'dual_nlm_high_filtereds.xlsx',
        'pickle_results_summary_high': '/workspace/data/output/set12/high_noisy/results/array_nln_high_filtereds.pkl',

        # Algorithmic parameters used internally by the experiment
        'f': 4,        # Patch radius
        't': 7,        # Search window radius    
        'h': 3.0,      # suavization parameter
        'alpha': 0.5,  # Geometric weight (for GEO-NLM)
    }

    # Execute the high-noise Gaussian experiment
    generate_gaussian_experiment_high_dual_nlm(parameters)
