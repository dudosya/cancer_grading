# Cancer Grading Project

This repository provides the code and instructions for reproducing the results from the cancer grading project using various deep learning models.

## Instructions for Reproducing the Code

### Step 1: Install Miniconda/Anaconda
Download and install Miniconda or Anaconda from their official website.

### Step 2: Clone the Repository
```bash
git clone https://github.com/dudosya/cancer_grading
cd cancer_grading
```

### Step 3: Create Conda Environment
- For CUDA-enabled GPU environment:
  ```bash
  conda env create -f environment.yml
  ```
- For CPU environment:
  ```bash
  conda env create -f environment_cpu.yml
  ```

### Step 4: Download the Dataset
1. Download the dataset from [KBSMC Colon Cancer Grading Dataset](https://drive.google.com/drive/folders/1tt8gEdIVRMJ0qsJzcPax6Di1u60WHFad?usp=sharing).
2. Unzip the dataset and place it in the repository folder so the structure looks like this:
   ```
   cancer_grading/
       main.py
       trainer.py
       models/
       ...
       data/
           tma_0x/
           Wsi_0x/
   ```

### Step 5: Modify Configuration
Edit `config.py` as needed to adjust the parameters. Choose the model in `main.py` by commenting or uncommenting the relevant lines:
```python
# Uncomment one of the following lines to select a model:

#myModel = cafenet.CaFeNet(num_classes=CONFIGURE.num_classes, lr=CONFIGURE.learning_rate).to(device=device)
#myModel = alexnet.AlexNet(num_classes=CONFIGURE.num_classes, lr = CONFIGURE.learning_rate).to(device=device)
#myModel = vgg16.VGG16(num_classes=CONFIGURE.num_classes, lr = CONFIGURE.learning_rate).to(device=device)
myModel = googlenet.GoogLeNet(num_classes=CONFIGURE.num_classes, lr = CONFIGURE.learning_rate).to(device=device)
#myModel = resnet34.ResNet34(num_classes=CONFIGURE.num_classes, lr = CONFIGURE.learning_rate).to(device=device)
#myModel = vit.VisionTransformer(num_classes=CONFIGURE.num_classes, lr = CONFIGURE.learning_rate).to(device=device)
#myModel = swin_tiny.SwinTiny(num_classes=CONFIGURE.num_classes, lr = CONFIGURE.learning_rate).to(device=device)
```

### Step 6: (Optional) Enable WandB Logging
Configure WandB in `config.py` as follows:
```python
wandb = True
wandb_project_name = "cancer_grading_tester1"
wandb_run_name = "test"
wandb_tags = ["test"]
wandb_monitor_gym = True
#wandb_run_id = "xvio3t2p"
#wandb_resume_run = "allow" # "never" otherwise
```
Set `wandb` to `False` if you do not wish to use WandB.

### Step 7: Run the Training
Execute the following command:
```bash
python main.py
```
