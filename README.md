# MetaGen

This is a code base for the paper "Inverse Design of Diffractive Metasurfaces Using Diffusion Models" by Hen et al.
[https://arxiv.org/abs/2506.21748](url)

 <img width="700" height="190" alt="metagen-scheme" src="https://github.com/user-attachments/assets/8541462f-3713-4e45-b7b5-f5049fea8400" />


---

## Installation

```bash
# Clone the repository
git clone https://github.com/liavhen/metagen.git
cd metagen
```
### Install dependencies
```bash
python -m venv metagen
source ./metagen/bin/activate
pip install -r requirements.txt
```

⚠️ Notice: This environment relies on CUDA or other GPU-compatible hardware.


### Environment Setup
1. Edit the global variables `STORAGE_DIR` and `PROJECT_DIR` in `utils/path.py` to match your machine. 
    --- `PROJECT_DIR` → path to the directory of the source files.
    --- `STORAGE_DIR` → path where logs, models, and data will be stored.
3. Run `utils/paths.py` directly to create the directory tree.
4. Download the desired models and datasets (see next section) into `ASSETS_DIR` and `DATA_DIR` respectively. 
Make sure paths are consistent with `data/data_config.py`.


### Models and Datasets
All datasets and models reported in the paper are shared through Hugging Face.
📂 [Datasets](https://huggingface.co/datasets/liavhen/metagen-datasets/tree/main)
🧩 [Models](https://huggingface.co/liavhen/metagen-models/tree/main)

## Running Scripts
Scripts for reproducing results from the paper are provided in:`./reproducibility_commands`.
These include examples for:
- Sampling trained models
- Performing gradient descent optimization for inverse design

### Create your own data
1. Configure your dataset in `data/data_config.py` by specifying the desired physical parameters.
2. Run `data/create_structures.py`.
  - For multi-wavelength datasets, use the arguments `start_at` and `end_at` o select wavelength indices (as ordered in `data_cfg.wavelengths`).

For convenience, a wrapper script `data/run_create_structures.py` runs all wavelengths in parallel on multiple GPUs.

### Train you own model
Train a new model with:
```
python diffusion/train.py --name <YOUR_EXP_NAME> --data_cfg <YOUR_DATA_CFG> --batch_size 16
```
Additional arguments are available for fine-grained control.
Refer to `diffusion/train.py` file for details.

## Citation
If you find our work useful, please consider citing it:
```
@article{
	hen2025metagen,
	author = {Hen, Liav and Yosef, Erez and Raviv, Dan and Giryes, Raja and Scheuer, Jacob},
	title = {Inverse Design of Diffractive Metasurfaces Using Diffusion Models},
	journal = {ACS Photonics},
	volume = {0},
	number = {0},
	pages = {null},
	year = {0},
	doi = {10.1021/acsphotonics.5c01384},
	URL = { https://doi.org/10.1021/acsphotonics.5c01384},
	eprint = { https://doi.org/10.1021/acsphotonics.5c01384}
}

```
