# CSE 586: Human Motion Prediction

This is a course project for PSU CSE 586: Computer Vision II. 

It encodes observed 3D human pose sequences (from the AMASS CMU subset) into dense representations using the [VPoser](https://github.com/nghorbani/human_body_prior) encoder, predicts future pose representations using an MLP or Transformer, and decodes these representations back to 3D poses using the VPoser decoder.

&nbsp;

## Prerequisites

### Environment

```bash
conda create -n 586 python
conda activate 586

pip install torch
pip install git+https://github.com/nghorbani/human_body_prior
pip install omegaconf loguru trimesh
```

### Data

Download:

- AMASS_CMUsubset
- VPoserModelFiles

&nbsp;

## Usage

Edit the folder and file paths in [`src/config.py`](src/configs.py) as needed.

Train:

```bash
python train.py
```

Evaluate:

```bash
python eval.py
```

Visualization: Open [`vis.ipynb`](./vis.ipynb) and run all the code.

&nbsp;

## Credits

- [siMLPe](https://github.com/dulucas/siMLPe)
- [makemore](https://github.com/karpathy/makemore)
