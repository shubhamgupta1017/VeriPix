import torch

# Model paths and names
# These variables define the architecture and the path for each model.
# The models are used for deepfake detection and super-resolution tasks.
deepfake_detection = "deepfake-detection"  # Name or path of the deepfake detection model
super_resolute = "super-resolute"  # Name or path of the super-resolution model

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu" 
print(f"Using device: {device}")  # Print the selected device

# Model weights file paths
# These paths point to the pre-trained weights for each model used in the tasks.
rrdb_weights_path = "models/weights/task_1/rrdb.pth"  # Path for RRDB model weights (used in super-resolution)
denseNet_weights_path = "models/weights/task_1/denseNet.pth"  # Path for DenseNet model weights (used in deepfake detection)

# Output file paths
# Define the path to save task-specific output, such as predictions or logs.
task1_output_file_path = "output/84_task1.json"  # Path for Task 1 output (e.g., deepfake detection results)

# Directories for data, temporary files, and output
# These paths define the locations where input data, temporary files, and output results are stored.
data_dir = "data"  # Directory for input images or datasets
temp_dir = "output/temp"  # Temporary directory for intermediate files
output_dir = "output"  # Directory for storing the final output (e.g., processed images, logs)
super_res_dir = "output/temp/super_res"  # Directory for storing super-resolved images
grad_cam_dir = "output/temp/grad_cam"  # Directory for storing Grad-CAM images

# Image format specification
# The format to be used when processing or saving images (e.g., jpg, png).
img_format = "png"  # Image format for input/output files
