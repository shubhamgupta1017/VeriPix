
# Adobe Mid Prep PS - Team 84 Submission

This contains the codebase for our submission to the **Adobe Mid Prep PS Competition**. The project focuses on **image classification** and **artifact detection**, leveraging advanced techniques in super-resolution, classification, and interpretability.

---

## Overview

### **Task 1: Fake Image Classification**
We use **Real-ESRGAN** for image super-resolution to enhance the quality of input images. The enhanced images are then classified using **DenseNet**, determining if the images are real or fake.

#### **Pipeline Steps**:
1. **Image Super-Resolution**: Apply **Real-ESRGAN** to upscale and enhance input images.
2. **Image Classification**: Use **DenseNet** for classifying enhanced images.

---

### **Task 2: Artifact Detection and Fix Explanation**
This task employs **Llama Vision 3.2 Instruct 11B** in combination with **Grad-CAM** boundary boxes (from the last layer of DenseNet). The pipeline identifies potential artifacts in images, narrows down possible issues, and generates explanations for artifact fixes.

#### **Pipeline Steps**:
1. **Artifact Detection**: Use Grad-CAM to localize areas of interest in the classified image.
2. **Fix Explanation**: Narrow artifact choices and provide explanations for the artifact fixes.

---

## Requirements

### **1. Install Dependencies**
- Install the required dependencies using:
  ```bash
  pip install -r requirements.txt
  ```
  *(This step can be skipped, as the token setup is pre-configured in `VLM.ipynb`.)*

### **2. (Optional token already added) Obtain Hugging Face Access Token**
- Log in or create an account at [Hugging Face](https://huggingface.co).
- Generate a new token from your [account settings](https://huggingface.co/settings/tokens).
- Save the token for configuring the pipeline.

Additionally:
- Visit [Meta](https://www.llama.com/llama-downloads/) to request access for **Llama Vision 3.2** models.
- Access [Llama 3.2 Vision Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) after gaining necessary permissions.

Configure Token in `VLM.ipynb`
- Paste the Hugging Face token into the appropriate section in `VLM.ipynb`.

---

## File Structure

Here’s an overview of the repository:

- **`main.py`**: The primary script for running the entire pipeline.
- **`data/`**: Contains input images for processing.
- **`models/`**: Stores downloaded model weights and the Real-ESRGAN model.
- **`output/`**: Contains outputs, including:
  - Results in `84_task1.json` and `84_task2.json`.
- **`densenet.py`**: Script for classifying images using DenseNet.
- **`super-resolution.py`**: Performs super-resolution using Real-ESRGAN.
- **`VLM.ipynb`**: Interactive notebook for Task 2.
- **`requirements.txt`**: Lists all required dependencies.

---

## Running the Pipeline

### **Step 1: Prepare Environment**
Install required dependencies by running:
```bash
pip install -r requirements.txt
```

### **Step 2: Place Input Files**
Add your input images to the `data/` folder.

### **Step 3: Execute Task 1**
Run the main pipeline for Task 1:
```bash
python main.py
```
- The first run may take longer due to model weight downloads.

### **Step 5: Execute Task 2**
Run the `VLM.ipynb` notebook to perform artifact detection and fix explanation.

### **Step 6: Review Outputs**
Results will be stored in the `output/` folder:
- **`84_task1.json`**: Results of image classification (real or fake).
- **`84_task2.json`**: Results of artifact detection and explanations.

Additional outputs include:
- **Super-resolved images**.
- **Grad-CAM visualizations**.

---

## Outputs

After executing the pipeline:
1. **Task 1 Output**:
   - `84_task1.json`: Contains classification results of input images.
   
2. **Task 2 Output**:
   - `84_task2.json`: Contains artifact detection and fix explanations.

