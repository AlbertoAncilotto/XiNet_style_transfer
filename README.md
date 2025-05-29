# XiNet Style Transfer - Training Notebook

This project provides a **PyTorch-based implementation** for training neural style transfer models using the **XiNet architecture**. The main training workflow is contained in `train_notebook.ipynb`, which demonstrates how to set up the dataset, configure training parameters, load models, compute losses, and run the training loop.

## Key Features
- Multiple style image options (see `images/` directory)
- Configurable training parameters for different styles
- Uses VGG, CLIP, and custom Transformer networks

## Usage
1. Install dependencies from `requirements.txt`.
2. Place your style images in the `images/` directory.
3. Adjust configuration cells in the notebook as needed.
4. Run `train_notebook.ipynb` to start training.

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- tqdm
- numpy
- COCO2017 dataset (or your own image dataset)

> For more details, see comments in the notebook cells.
