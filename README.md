# Neural network for wing lineage classification (C / M / O)

All results below were obtained on the `Normalized_PNG` folder: 665 images from 72 colonies (C: 247, M: 130, O: 288). Compared with the previous version, seven problematic images were removed and one M image was corrected. The old results (672 BMP images) are kept in the `archive_bmp_672/` folder.

## Results on the fold 0 test set (20%, 129 images from 15 colonies)
- Per-wing accuracy: **93.8%** (121 of 129), macro F1 = 0.930
- Per-colony accuracy (majority vote of the wings in each colony): **100%** (15 of 15)

| Lineage | Precision | Recall | Test images |
|---|---|---|---|
| C | 0.943 | 0.980 | 51 |
| M | 0.947 | 0.857 | 21 |
| O | 0.930 | 0.930 | 57 |

Confusion matrix (rows = true, columns = predicted, order C, M, O):
```
[[50  0  1]
 [ 0 18  3]
 [ 3  1 53]]
```

## 5-fold cross-validation grouped by colony
| Method | Wing accuracy | Macro F1 | M recall | Colonies correct |
|---|---|---|---|---|
| Pretrained ResNet18 (ImageNet) | 92.9 ± 4.1% | 0.919 | 0.831 | 71/72 |
| Small CNN from scratch | 91.7 ± 4.2% | 0.912 | 0.869 | 69/72 |
| PCA + LDA | 90.2 ± 1.9% | 0.889 | 0.808 | 72/72 |
| HOG + SVM | 88.4 ± 2.4% | 0.865 | 0.685 | 70/72 |

McNemar's test with Holm correction: only the difference between ResNet18 and HOG + SVM is significant (p = 0.0012; 0.007 after correction).

## Data and split
- 80/20 split, stratified by lineage and grouped by colony number (the number before the first `-` in the filename), so no colony appears in both training and test.
- Fold 0: 431 training, 105 validation (11 colonies), 129 test (15 colonies). The exact file lists are in `results/split.json`.

## Model
- A small CNN (5 convolutional blocks, about 1.5 million parameters) trained from scratch; grayscale 320×144 input.
- Data augmentation: ±4° rotation, small shifts and scaling, brightness and contrast changes.
- AdamW, 60 epochs, class weighting to compensate for the smaller M class.

## Running on your own computer
```
pip install torch torchvision scikit-learn pillow
python train_wing_cnn.py --data "E:\ML\subspecies\subspecies Train\Normalized_PNG" --out results
```
GPU version (much faster on an NVIDIA card; requires the CUDA build of PyTorch):
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python train_wing_cnn_gpu.py --data "E:\ML\subspecies\subspecies Train\Normalized_PNG" --out results
```
- ResNet18: `--pretrained --weights resnet18_a1_0-d63eafa0.pth --epochs 30`
  (weights from github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth)
- Other cross-validation folds: `--fold 0..4`

## Files
- `train_wing_cnn.py` (CPU), `train_wing_cnn_gpu.py` (GPU): training code
- `results/`: fold 0 model (`best_model.pt`), test report, per-image predictions, training history (`history.json`, `run.log`)
- `experiments/`: `baseline_classic.py`, `gradcam.py`, `summarize_cv.py`, `run_all.sh` and the `gradcam_fold0.png` map
- `experiments/cv_results/`: per-fold results for the CNN (folds 1–4; fold 0 is `results/`) and ResNet18, the classic methods, and the `cv_summary.json` summary

## Limitations
There are only 72 colonies (15 of them M); the standard deviation of about 4% across folds shows that each number is approximate.
