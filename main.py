# Import necessary modules and configurations from a separate config file
from config import *  
import os   
import importlib 
from PIL import Image 
from torchvision import transforms  
import torch.backends.mkldnn as mkldnn  
mkldnn.enabled = True  
import time
# Dynamically load the necessary functions and classes from external modules
# - SuperResolution model for enhancing image resolution
# - DenseNet model for deepfake detection
# - GradCAM model for generating class activation maps for interpretability
SuperResolution = getattr(importlib.import_module('super-resolution'), 'SuperResolution')
DenseNet = getattr(importlib.import_module('densenet'), 'DenseNet')
GradCAM = getattr(importlib.import_module('densenet'), 'GradCAM')


# ------------------------ Model Initialization ------------------------
# Initialize the models used for deepfake detection, super-resolution, and Grad-CAM
deepfake_detection_model = DenseNet()  # Load the DenseNet model for deepfake detection
super_resolution_model = SuperResolution()  # Load the super-resolution model
grad_cam_model = GradCAM()  # Load the Grad-CAM model for class activation mapping

# -------------------- Part 1: Deepfake Detection --------------------

# Loop through all images in the data directory for processing
for index in range(1, len(os.listdir(data_dir)) + 1):
    # Generate the path to the input image file
    input_image_path = f"{data_dir}/{index}.{img_format}"
    
    # Open and convert the image to RGB format
    input_image = Image.open(input_image_path).convert('RGB')
    
    # Convert the image to a tensor, and add a batch dimension
    input_image_tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(torch.device(device))
    
    # Apply super-resolution to the input image tensor to enhance its quality
    super_res_tensor = super_resolution_model.super_resolve_tensor(input_image_tensor)
    
    # Save the enhanced (super-resolved) image using the super-resolution model
    super_resolution_model.save_super_res_image(super_res_tensor, index)
    
    # Perform deepfake detection prediction on the enhanced image
    prediction = deepfake_detection_model.predict(index)

# After processing all images, save the results of the deepfake detection
deepfake_detection_model.add_results_to_json()

# -------------------- Part 2: Grad-CAM --------------------

# Loop through all images again for generating Grad-CAM results
for index in range(1, len(os.listdir(data_dir)) + 1):
    # Generate and save the Grad-CAM visualization for the current image
    grad_cam_model.save_gradCAM(index)

