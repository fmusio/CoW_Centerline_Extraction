import sys
import os
# add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import subprocess
import shutil
from configs import nnunet_results

from logger import logger

def nnunet_predict_skeleton(mlt_mask_filepath: str, nnunet_dir: str, predicted_skeleton_dir: str, dataset_id: str = '113', folds: str = 'all', trainer: str = 'nnUNetTrainer', 
                            configuration: str = '3d_fullres', plans: str = 'nnUNetPlans'):
    """
    Run nnUNet inference to predict a skeleton from a given mask file.
    This function processes a mask file through the nnUNet deep learning framework
    to predict a skeletal structure. It executes the nnUNet inference via a bash script, and cleans up
    temporary files afterward.
    Args:
        mlt_mask_filepath (str): Path to the input mask file to be processed.
        nnunet_dir (str): Base directory for nnUNet operations.
        predicted_skeleton_dir (str): Directory where predicted skeleton outputs will be saved.
        dataset_id (str, optional): Dataset ID for nnUNet. Defaults to '113'.
        folds (str, optional): Folds to use for prediction. Defaults to 'all'.
        trainer (str, optional): nnUNet trainer to use. Defaults to 'nnUNetTrainer'.
        configuration (str, optional): nnUNet configuration. Defaults to '3d_fullres'.
        plans (str, optional): nnUNet plans. Defaults to 'nnUNetPlans'.
    Returns:
        str: Path to the predicted skeleton file.
    Note:
        This function requires a properly configured nnUNet installation and
        expects the inference.sh bash script to be present in the nnunet_dir.
    """
    nnunet_input_dir = os.path.join(nnunet_dir, 'input')
    # create directionaries
    if not os.path.exists(nnunet_input_dir):
        logger.debug(f"Creating directory: {nnunet_input_dir}")
        os.makedirs(nnunet_input_dir, exist_ok=True)
    
    # delete previous input files
    for file in os.listdir(nnunet_input_dir):
        file_path = os.path.join(nnunet_input_dir, file)
        if os.path.isfile(file_path):
            logger.debug(f"Removing previous input file: {file_path}")
            os.remove(file_path)
    
    if not os.path.exists(predicted_skeleton_dir):
        logger.debug(f"Creating directory: {predicted_skeleton_dir}")
        os.makedirs(predicted_skeleton_dir, exist_ok=True)

    logger.info(f'Running nnUNet inference with args:'
                f'\n\t- input file={mlt_mask_filepath}'
                f'\n\t- nnunet_dir={nnunet_dir}'
                f'\n\t- output file={predicted_skeleton_dir}'
                f'\n\t- dataset_id={dataset_id}'
                f'\n\t- folds={folds}'
                f'\n\t- trainer={trainer}'
                f'\n\t- configuration={configuration}'
                f'\n\t- plans={plans}'
               )
    
    logger.info(f"Using nnUNet_results from environment variable: {os.environ.get('nnUNet_results', 'Not Set')}")

    # copy input mlt_mask_file to nnunet input directory
    logger.info(f"Copying {mlt_mask_filepath} to {nnunet_input_dir}...")
    shutil.copy(mlt_mask_filepath, nnunet_input_dir)

    bash_file = os.path.join(nnunet_dir, 'inference.sh')

    # Prepare command with individual fold arguments
    command = [
        "bash",
        bash_file,
        dataset_id,
        nnunet_input_dir,
        predicted_skeleton_dir,
        folds,
        trainer,
        configuration,
        plans,
    ]

    logger.debug(f"Calling nnUNet_pred with subprocess command {command}...")

    # Run nnUNet inference with output capture
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    # Log output in real-time
    for line in iter(process.stdout.readline, ''):
        logger.info(line.strip())
        
    # After stdout is done, check for any errors
    for line in iter(process.stderr.readline, ''):
        logger.error(line.strip())

    process.wait()

    pred_skel_file = os.path.join(predicted_skeleton_dir, os.path.basename(mlt_mask_filepath).replace('_0000.nii.gz', '.nii.gz'))

    if os.path.exists(pred_skel_file):
        logger.info(f"Predicted skeleton saved to {predicted_skeleton_dir}.")
        # rename file if it contains '_connected' in the name
        if pred_skel_file.endswith('_connected.nii.gz'):
            pred_skel_file_saved = pred_skel_file.replace('_connected.nii.gz', '.nii.gz')
            os.rename(pred_skel_file, pred_skel_file_saved)
            pred_skel_file = pred_skel_file_saved

    else:
        logger.error(f"Predicted skeleton not found in {predicted_skeleton_dir}. Please check the nnUNet inference process.")
        raise FileNotFoundError(f"Predicted skeleton not found in {predicted_skeleton_dir}. Please check the nnUNet inference process.")

    # Remove file in input directory
    logger.debug(f"Removing temp file {os.path.basename(mlt_mask_filepath)} from {nnunet_input_dir}...")
    os.remove(os.path.join(nnunet_input_dir, os.path.basename(mlt_mask_filepath)))
    logger.info(f"nnUNet inference completed for {mlt_mask_filepath}.")
    
    return pred_skel_file