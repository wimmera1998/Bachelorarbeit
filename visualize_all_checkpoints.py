import os
import subprocess

def visualize_checkpoints(checkpoint_dir, vis_pred_path, config_path, thresholds):
    """
    Visualizes every 4th model checkpoint in a directory and the last checkpoint.

    Parameters:
    - checkpoint_dir: The directory where model checkpoints are stored.
    - vis_pred_path: The path to the visualization script (vis_pred.py).
    - config_path: The path to the configuration file.
    - thresholds: A list of thresholds to use for visualization.
    """
    
    # List all checkpoints (e.g., epoch_13.pth)
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth') and 'epoch_' in f]
    
    # Sort checkpoints by epoch number
    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # Get the total number of checkpoints
    num_checkpoints = len(checkpoints)
    
    for idx, checkpoint in enumerate(checkpoints):
        epoch_num = int(checkpoint.split('_')[1].split('.')[0])
        model_path = os.path.join(checkpoint_dir, checkpoint)
        
        # Only visualize every 4th checkpoint and the last one
        if epoch_num % 24 == 0 or idx == num_checkpoints - 1:
            # Create the directory for visualization results of the current epoch
            show_dir = os.path.join(checkpoint_dir, f'vis_pred/epoch_{epoch_num}_visuals')
            os.makedirs(show_dir, exist_ok=True)
            
            # Loop through the defined thresholds
            for thresh in thresholds:
                # Command to execute the visualization script
                vis_command = f'python {vis_pred_path} {config_path} {model_path} --score-thresh {thresh} --show-dir {show_dir}'
                
                # Output the command for verification
                print(f"Running: {vis_command}")
                
                # Execute the visualization script
                try:
                    subprocess.run(vis_command, shell=True, check=True)
                    print(f"Visualization completed for epoch {epoch_num} with --score-thresh {thresh}: {show_dir}")
                except subprocess.CalledProcessError as e:
                    print(f"Error running visualization script for epoch {epoch_num} with --score-thresh {thresh}: {e}")

if __name__ == "__main__":
    # Define the path to the model checkpoints
    checkpoint_dir = "/disk/vanishing_data/qo691/MapTR/work_dirs/maptr_tiny_r50_24e_cwl_fusion_b4_w4_a8_XU_PreTra_Re_Sk_bcwl_1"
    
    # Path to the visualization script
    vis_pred_path = "/disk/no_backup/qo691/MapTR/tools/maptr/vis_pred.py"
    
    # Path to the configuration file
    # Important to use a configuration file with the test file is picked the right.
    # test=dict(type=dataset_type,
    #          data_root=data_root,
    #          aerialdata_root=aerialdata_root,
    #          ann_file=data_root + 'nuscenes_infos_5_test_samples.pkl', #nuscenes_infos_5_test_samples.pkl

    config_path = "/disk/no_backup/qo691/MapTR/projects/configs/maptr/maptr_tiny_r50_24e_adrian2.py"

    
    
    # Define thresholds
    thresholds = [0.0, 0.2, 0.3, 0.4]
    
    # Start the visualization of every 4th checkpoint and the last one
    visualize_checkpoints(checkpoint_dir, vis_pred_path, config_path, thresholds)
