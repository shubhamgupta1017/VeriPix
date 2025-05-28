import os
import torch
from torchvision import transforms
from PIL import Image
from config import *  
from models.RealESRGAN import RealESRGAN 

class SuperResolution:
    """
    A class that performs super-resolution on low-resolution images using RealESRGAN.
    """
    
    def __init__(self):
        """
        Initializes the SuperResolute class with the necessary weights for the model.
        
        Args:
        """
        # Ensure the output directory exists, create if it doesn't
        os.makedirs(f"{super_res_dir}", exist_ok=True)
        # Set the device (GPU or CPU) for running the model
        self.device = torch.device(device)  # 'device' is defined in your config or system

        # Initialize the RealESRGAN model with a scaling factor of 4 (4x resolution increase)
        super_res_model = RealESRGAN(self.device, scale=4)
        # Access and load the RRDB model (Residual in Residual Dense Block), which performs the core super-resolution task
        self.rrdb_model = super_res_model.model.to(self.device)  # Move the model to the specified device (GPU or CPU)
        self.rrdb_model.load_state_dict(torch.load(rrdb_weights_path, map_location=self.device))
        
        # Set the model to evaluation mode to disable training-specific operations like dropout
        self.rrdb_model.eval()

    def super_resolve_tensor(self, low_res_image_tensor):
        """
        Applies super-resolution to a low-resolution image tensor.
        
        Args:
            low_res_image_tensor (torch.Tensor): A tensor representing the low-resolution input image.
        
        Returns:
            torch.Tensor: The super-resolved image as a tensor.
        """
        # Ensure the low-resolution image tensor is on the correct device (GPU/CPU)
        low_res_image_tensor = low_res_image_tensor.to(self.device)
        
        with torch.no_grad():  # Disable gradient computation as we're only doing inference
            # Perform the super-resolution using the RRDB model
            super_res_image_tensor = self.rrdb_model(low_res_image_tensor)
            super_res_image_tensor = torch.clamp(super_res_image_tensor.squeeze(0).cpu(), 0, 1)  # Clamp values to [0, 1]
            # Return the super-resolved image tensor back to CPU for further processing (if needed)
            return super_res_image_tensor.cpu()
    
    def save_super_res_image(self, super_res_tensor, index):
        """
        Converts the super-resolved tensor to an image and saves it to the disk.
        
        Args:
            super_res_tensor (torch.Tensor): The tensor representing the super-resolved image.
            index (int): The index for naming the output image file.
        """
        # Convert the super-resolution tensor to a PIL Image for saving
        super_res_image = transforms.ToPILImage()(super_res_tensor)
        
        
        # Save the super-resolved image to the specified path
        super_res_image.save(f"{super_res_dir}/{index}.png")

        
if __name__ == "__main__":
    """
    Main function to load a low-resolution image, apply super-resolution, and save the result.
    """
    index=1
    # Path to the low-resolution image
    image_path = f"data/{index}.{img_format}"
    
    # Load the low-resolution image using PIL
    low_res_image = Image.open(image_path)
    
    # Convert the loaded image to a tensor and ensure it's on the correct device (GPU or CPU)
    low_res_image_tensor = transforms.ToTensor()(low_res_image).unsqueeze(0).to(device)
    # Initialize the SuperResolute class with the paths to the model weights
    super_resolute = SuperResolution()
    # Apply super-resolution to the low-resolution image tensor
    super_res_tensor = super_resolute.super_resolve_tensor(low_res_image_tensor)
    
    # Save the super-resolved image to the disk
    super_resolute.save_super_res_image(super_res_tensor, index)  # Saving the  image
    
    # Print a message indicating the process is complete
    print("Super resolution completed and saved.")
