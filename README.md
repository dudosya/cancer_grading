# Application of CNNs and Vision Transformers in Cancer Grading

This repository contains the official implementation for the capstone project report, **"Application of Convolutional Neural Networks and Vision Transformers in Cancer Grading in Pathology Images."**

[![Paper](https://img.shields.io/badge/Read%20the%20Full-Paper-blue)](https://nur.nu.edu.kz/handle/123456789/8849)

## Abstract

Cancer grading is a time-consuming and labor-intensive process. While existing automated methods use Convolutional Neural Networks (CNNs) with attention modules like the Convolutional Block Attention Module (CBAM) for feature refinement, their conventional integration can be suboptimal. Integrating CBAM sequentially between convolutional blocks can disrupt information flow, increase model parameters, and negatively impact performance.

This project proposes a novel **Post-Convolutional Parallel CBAM** architecture. Instead of sequential placement, our method places a parallel version of CBAM after the main feature extraction backbone, serving as a final refinement step before classification. We evaluated this approach on the KBSMC colon cancer dataset using VGG16, GoogLeNet, and ResNet34 architectures. The results demonstrate that our proposed method consistently outperforms both baseline models and models using the original sequential CBAM integration. For instance, our approach achieved an F1-score of 0.810 compared to 0.658 from the original CBAM method on the same backbone, despite having fewer parameters. This demonstrates a more effective and efficient approach for transfer learning in medical image analysis.

### The Paper

The complete methodology, experiments, and results are detailed in the full report, officially published in the Nazarbayev University Repository.

> Erlan, D. (2025). *Application of Convolutional Neural Networks and Vision Transformers in Cancer Grading in Pathology Images*. Nazarbayev University School of Engineering and Digital Sciences.

**[Read the Full Paper Here (Permanent Link)](https://nur.nu.edu.kz/handle/123456789/8849)**

### Keywords
`Cancer Grading`, `Deep Learning`, `Convolutional Neural Networks`, `Attention Modules`, `Medical Image Analysis`, `Pathology Images`

---

## Instructions for Reproducing the Code

### Step 1: Install Miniconda/Anaconda
Download and install Miniconda or Anaconda from their official website if you haven't already.

### Step 2: Clone the Repository
```bash
git clone https://github.com/dudosya/cancer_grading
cd cancer_grading
```

### Step 3: Create Conda Environment
A Conda environment ensures that all dependencies are correctly installed.

- **For systems with a CUDA-enabled GPU:**
  ```bash
  conda env create -f environment.yml
  conda activate cancer-grading
  ```
- **For CPU-only systems:**
  ```bash
  conda env create -f environment_cpu.yml
  conda activate cancer-grading-cpu
  ```

### Step 4: Download the Dataset
1. Download the dataset from the [KBSMC Colon Cancer Grading Dataset](https://drive.google.com/drive/folders/1tt8gEdIVRMJ0qsJzcPax6Di1u60WHFad?usp=sharing).
2. Unzip the dataset and place the `tma_0x` and `Wsi_0x` folders inside a `data` directory at the root of the repository. The final structure should look like this:
   ```
   cancer_grading/
   ├── main.py
   ├── trainer.py
   ├── models/
   ├── data/
   │   ├── tma_0x/
   │   └── Wsi_0x/
   └── ...
   ```

### Step 5: Configure Your Experiment
All training hyperparameters (learning rate, batch size, number of epochs) can be adjusted in the `config.py` file.

### Step 6: Run the Training
The `main.py` script is organized into functions, where each function runs an experiment with a specific model architecture using our proposed Post-Convolutional Parallel CBAM.

1.  Open `main.py`.
2.  Navigate to the bottom of the file (`if __name__ == "__main__":`).
3.  Comment or uncomment the experiment functions to select which model to train.

The available experiments are:
*   `exp1()`: Trains a custom AlexNet model.
*   `exp2()`: Trains a custom ResNet34 model.
*   `exp3()`: Trains a custom VGG16 model.

**Example: To run the training for ResNet34:**
```python
if __name__ == "__main__":
    # exp1()  # Run AlexNet
    exp2()    # Run ResNet34
    # exp3()  # Run VGG16
```

4.  Execute the script from your terminal:
    ```bash
    python main.py
    ```

### Step 7: (Optional) Enable Weights & Biases Logging
This project supports logging metrics and results to [Weights & Biases](https://wandb.ai).

1.  Open `config.py`.
2.  Set `wandb = True`.
3.  Configure your project details:
    ```python
    wandb = True
    wandb_project_name = "cancer_grading_final"
    wandb_run_name = "resnet34-parallel-cbam"
    wandb_tags = ["ResNet34", "Proposed-Method"]
    wandb_monitor_gym = True
    # For resuming a run, uncomment and set the following:
    # wandb_run_id = "your_run_id"
    # wandb_resume_run = "allow"
    ```
4.  If you set `wandb = False`, logging will be disabled.

---

## Citation
If you find this work useful in your research, please consider citing the report:
```bibtex
@mastersthesis{Erlan2025CancerGrading,
  author       = {Erlan, Dosbol},
  title        = {Application of Convolutional Neural Networks and Vision Transformers in Cancer Grading in Pathology Images},
  school       = {Nazarbayev University School of Engineering and Digital Sciences},
  year         = {2025},
  month        = {May},
  note         = {Bachelor's Thesis},
  url          = {https://nur.nu.edu.kz/handle/123456789/8849}
}
```
