# Swin Transformer for Tiny ImageNet Classification

This repository contains an implementation of the Swin Transformer architecture for image classification on the Tiny ImageNet dataset. The implementation is based on the architecture from the original Swin Transformer paper, with optimizations for better resource utilization.

## Dataset

Tiny ImageNet contains 100,000 images of 200 classes (500 for each class) downsized to 64×64 colored images:
- Each class has 500 training images, 50 validation images, and 50 test images
- The dataset is a subset of the ImageNet dataset

## Project Structure

The project is organized into three main Python files:

- `model.py`: Contains the complete Swin Transformer architecture implementation
- `data.py`: Handles data loading and augmentation for Tiny ImageNet
- `train.py`: Implements the training and evaluation pipeline

## Requirements

- PyTorch >= 1.7.0
- torchvision >= 0.8.0
- timm >= 0.4.12
- numpy
- tqdm
- PIL (Pillow)

## Usage

### Training

To train the standard Swin-T model:

```bash
python train.py --batch-size 128 --output-dir output/swin_t
```

To train the efficient variant with fewer parameters:

```bash
python train.py --efficient --batch-size 128 --output-dir output/swin_t_efficient
```

To include mixup and cutmix augmentations:

```bash
python train.py --mixup --batch-size 128 --output-dir output/swin_t_mixup
```

### Resuming Training

To resume training from a checkpoint:

```bash
python train.py --resume output/swin_t/checkpoint_epoch10.pth
```

### Evaluation

To evaluate a trained model:

```bash
python train.py --evaluate --resume output/swin_t/model_best.pth
```

## Model Architecture

The implementation follows the Swin-T architecture from the original paper:

- Patch Size: 4×4
- Embedding Dimension: 96
- Depths: [2, 2, 6, 2]
- Number of Heads: [3, 6, 12, 24]
- Window Size: 7×7

The efficient variant uses:
- Embedding Dimension: 64
- Depths: [2, 2, 4, 2]
- Number of Heads: [2, 4, 8, 16]

## Key Features

1. **Hierarchical Feature Learning**: The model uses a hierarchical transformer with patch merging to downsample feature maps.
2. **Shifted Window Self-Attention (SW-MSA)**: Efficient local attention with shifted windows for capturing global dependencies.
3. **Advanced Data Augmentation**: Integration of mixup, cutmix, and RandAugment techniques.
4. **Cosine Learning Rate Schedule**: With warmup to stabilize early training.
5. **Parameter Efficiency**: The efficient variant reduces parameters while maintaining performance.

## Model Performance

- Standard Swin-T: ~28M parameters
- Efficient Swin-T: ~14M parameters

## Command-line Arguments

Important command-line arguments for `train.py`:

| Argument           | Description                              | Default       |
|--------------------|------------------------------------------|---------------|
| `--efficient`      | Use efficient model variant              | False         |
| `--batch-size`     | Batch size for training                  | 128           |
| `--epochs`         | Number of training epochs                | 100           |
| `--lr`             | Learning rate                            | 5e-4          |
| `--mixup`          | Enable mixup and cutmix augmentation     | False         |
| `--output-dir`     | Directory to save checkpoints            | "output"      |
| `--resume`         | Path to checkpoint for resuming training | ""            |
| `--evaluate`       | Run evaluation only                      | False         |

## Troubleshooting

### Data Loading Error

If you encounter an error related to image loading and timm's auto augmentation:

```
TypeError: 'builtin_function_or_method' object is not subscriptable
```

This occurs because of an incompatibility between PyTorch tensor image representations and timm's augmentation functions that expect PIL images. The fix is to:

1. Make sure to use PIL images (`PIL.Image.open()`) rather than PyTorch tensors (`torchvision.io.read_image()`) in the data loading pipeline
2. Don't include `transforms.ToPILImage()` when the input is already a PIL image

### Meshgrid Warning

If you see a warning about `torch.meshgrid`:

```
UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.
```

Fix this by specifying the indexing parameter in all torch.meshgrid calls:

```python
coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
```

## License

This project is released under the MIT License. 