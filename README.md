# VPoser Environment Setup for Linux Compilation

This guide will help you set up the **VPoser environment** using Conda and Pip if you decide to compile it on a Linux machine and not use Google Colab. Follow these steps to ensure a smooth and efficient installation.

---

### Prerequisites

Make sure you have the following installed on your system:

- **Conda** (Miniconda3 or Anaconda)
- **Python** (version 3.x)

---

### Setup Instructions

1. **Create a Conda Environment**

   ```bash
   conda create -n vposer python
   ```

   This will create a new Conda environment named `vposer`.

2. **Activate the Environment**

   ```bash
   conda activate vposer
   ```

   Ensure that the environment is active before proceeding to the next steps.

3. **Install Human Body Prior Package and other packages**

   Clone and install the **Human Body Prior** library:
   ```bash
   pip install git+https://github.com/nghorbani/human_body_prior
   pip install omegaconf loguru trimesh
   pip3 install torch
   ```
---

### Verifying the Installation

Run the following command to check the PyTorch installation:
```bash
python -c "import torch; print(torch.__version__)"
```
You should see the installed version of **PyTorch** displayed.

---

### Directory Structure
Download all the required files from the Canvas project page and place them in the 
root folder VPoser. The folder structure should be as shown below 
```
VPOSER/   
├── ReadMe.md             # This README file
├── testVPoser.ipynb      # Jupyter notebook to test VPoser
└── VPoserModelFiles/     # Contains pre-trained model files
```

### Changes to `testVPoser.ipynb` File
Remove the following blocks from your `testVPoser.ipynb` file since we don't need it
any more

```bash
# compile/install code that is needed for this demo
# if it says that you need to restart the runtime, go ahead and do that,
# then run this step again to make sure the installs are complete

!pip install git+https://github.com/nghorbani/human_body_prior
!pip install omegaconf
!pip install loguru
!pip install trimesh
```

```bash
# connect to your google drive
# alternatively, upload the VPoserTest directory/files into Colab

from google.colab import drive
drive.mount('/content/gdrive')
```


Make the following path change in your `testVPoser.ipynb` file
```bash
support_dir = './VPoserModelFiles/'
```



---

### Troubleshooting

- If you encounter errors during installation, try updating `pip`:
  ```bash
  pip install --upgrade pip
  ```
- Make sure that the **Conda environment is active** before running the installation commands.

---

### You're All Set!

You have successfully set up the **VPoser environment**. You can now start using VPoser and related tools within this environment.
