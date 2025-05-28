import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
from torchvision import models, transforms
import json
from config import *

# Define a DenseNet class that will be used for binary image classification
class DenseNet:
    def __init__(self):
        # Initialize an empty list to store results of predictions
        self.results = []
        
        # Load a pre-trained DenseNet model without weights
        self.model = models.densenet121(weights=None)
        
        # Modify the classifier to have a single output unit for binary classification
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier.in_features, 1),   # Change the output to 1 unit
            nn.Sigmoid()                          # Sigmoid activation for binary classification
        )

        # Load pre-trained weights from a specified file path
        self.model.load_state_dict(torch.load(denseNet_weights_path, map_location=device))
        self.model = self.model.to(device)  # Move the model to the specified device (GPU or CPU)
        self.model.eval()  # Set the model to evaluation mode
        
        # Define the transformation pipeline for the input images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to match expected input size
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ImageNet
        ])
        
    # Function to load an image and apply the necessary transformations
    def load_image(self, image_path):
        img = Image.open(image_path).convert('RGB')  # Open the image and convert to RGB
        img_tensor = self.transform(img).unsqueeze(0).to(device)  # Apply the transformations and add batch dimension
        return img_tensor

    # Function to make a prediction for an image by index
    def predict(self, index):
        image_path = f"{super_res_dir}/{index}.{img_format}"  # Generate the image path based on index
        image_tensor = self.load_image(image_path)  # Load and transform the image
        output = self.model(image_tensor)  # Get the model's output
        prediction = "real" if output < 0.5 else "fake"  # Use threshold for binary classification
        self.add_to_results(index, prediction)  # Add the result to the results list
        return prediction
    
    # Function to add prediction results to the results list
    def add_to_results(self, index, prediction):
        self.results.append({"index": index, "prediction": prediction})
    
    # Function to save the prediction results to a JSON file
    def add_results_to_json(self):
        with open(task1_output_file_path, "w") as f:
            json.dump(self.results, f, indent=4)  # Write results as JSON with indentation

# Define a GradCAM class that extends DenseNet to include Grad-CAM visualization
class GradCAM(DenseNet): 
    def __init__(self):
        super().__init__()  # Initialize the parent DenseNet class
        os.makedirs(grad_cam_dir, exist_ok=True)  # Create the directory for Grad-CAM output if it doesn't exist

    # Function to register hooks for capturing activations and gradients
    def register_hooks(self):
        self.activations = []  # List to store activations
        self.gradients = []  # List to store gradients

        # Register a hook on the last convolutional layer to capture activations and gradients
        last_conv_layer = self.model.features.denseblock4.denselayer12.conv2  # Last convolutional block in DenseNet
        last_conv_layer.register_forward_hook(self.hook_activation)  # Hook for activations
        last_conv_layer.register_full_backward_hook(self.hook_gradient)  # Hook for gradients

        return self.activations, self.gradients

    # Hook to capture activations during forward pass
    def hook_activation(self, module, input, output):
        self.activations.append(output)  # Store activations in the list

    # Hook to capture gradients during backward pass
    def hook_gradient(self, module, grad_in, grad_out):
        self.gradients.append(grad_out[0])  # Store gradients in the list
    
    # Function to generate Grad-CAM heatmap for a given image
    def gradcam_heatmap(self, image_tensor):
        self.activations, self.gradients = self.register_hooks()  # Register hooks and get activations and gradients
        output = self.model(image_tensor)  # Perform forward pass
        self.model.zero_grad()  # Zero the gradients
        
        output = output.squeeze()  # Remove unnecessary dimensions
        output.backward()  # Perform backward pass to get gradients
    
        self.gradients = self.gradients[0]  # Get the first (and only) gradient
        self.activations = self.activations[0]  # Get the first (and only) activation
        
        # Calculate the weight for each activation map
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Calculate the Grad-CAM heatmap by weighted sum of activations
        heatmap = torch.mean(self.activations * weights, dim=1).squeeze().cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)  # Set negative values to zero
        heatmap = cv2.resize(heatmap, (image_tensor.shape[2], image_tensor.shape[3]))  # Resize heatmap to image size
        max_val = np.max(heatmap)  # Find the maximum value in the heatmap
        if max_val != 0:
            heatmap /= max_val  # Normalize the heatmap
        else:
            heatmap = np.zeros_like(heatmap)  # Handle case where max value is zero
        return heatmap
    
    # Function to save the Grad-CAM visualization with bounding box
    def save_gradCAM(self, index):
        image_path = f"{super_res_dir}/{index}.{img_format}"  # Generate image path based on index
        image_tensor = self.load_image(image_path)  # Load and transform the image
        heatmap = self.gradcam_heatmap(image_tensor)  # Generate Grad-CAM heatmap
        
        img = Image.open(image_path).convert('RGB')  # Open the image
        img = np.array(img)  # Convert the image to a NumPy array
        
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # Resize heatmap to match image size
        heatmap = np.clip(heatmap_resized, 0, 1)  # Clip heatmap values to the range [0, 1]
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)  # Apply color map to heatmap
        
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)  # Convert heatmap to RGB format
        heatmap_rgb = heatmap_rgb.astype(np.float32) / 255.0  # Normalize the heatmap to the range [0, 1]
        superimposed_img = heatmap_rgb * 0.4 + img / 255.0  # Superimpose heatmap on the image

        # Find the maximum activation point for bounding box
        max_loc = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
        y, x = max_loc
        
        # Draw a bounding box around the maximum activation point
        half_box_size = 40  # Size of the bounding box
        x1, x2 = max(0, x - half_box_size), min(img.shape[1], x + half_box_size)
        y1, y2 = max(0, y - half_box_size), min(img.shape[0], y + half_box_size)
        
        cv2.rectangle(superimposed_img, (x1, y1), (x2, y2), color=(1, 0, 0), thickness=2)  # Draw rectangle
        
        # Convert back to PIL Image for saving
        superimposed_img = np.clip(superimposed_img * 255, 0, 255).astype('uint8')
        pil_image = Image.fromarray(superimposed_img)

        # Save the image with Grad-CAM visualization
        pil_image.save(f'{grad_cam_dir}/{index}.{img_format}')

# Main code to run Grad-CAM on a set of images
if __name__ == "__main__":
    gradcam = GradCAM()  # Instantiate the GradCAM class
    index=1
    gradcam.save_gradCAM(index)  # Generate and save Grad-CAM visualization for each image
