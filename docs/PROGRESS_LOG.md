
# Progress Log

## 9–13 November 2026

Updated on 12 November.

- Completed the first EDA iteration, awaiting teacher feedback.
- Analysed the OpenEarthMap dataset, identifying 2,687 valid image-mask pairs and missing data.
- Explored class distributions, geographical representation and image-mask consistency.
- Identified class imbalance, especially for Bareland and Water, and planned to ignore Unknown pixels during training and evaluation.
- Documented the main findings and decisions to guide preprocessing and modelling.
- Set up the GitHub repository and established the project timeline.

### Next Steps

- Improve the EDA based on teacher feedback and complete data validation.
- Prepare geographical splits and preprocessing.
- Develop and evaluate the first CNN baseline.

## 16–20 November 2026

Updated on 18 November.

- Reviewed preprocessing validation for 2,687 image-mask pairs across 75 regions, including readability, dimensions, RGB channels and mask label IDs.
- Confirmed that the recorded duplicate checks found no exact duplicate images or pairs.
- Verified the saved geographical splits: 1,942 training pairs, 275 validation pairs and 470 test pairs, with no region overlap and all eight classes represented in each split.
- Prepared resizing to 512 × 512 pixels, image scaling to 0–1 and label conversion from 1–8 to 0–7, with Unknown pixels marked as 255.
- Validated the reusable preprocessing function and one PyTorch DataLoader batch from each split, checking shapes, data types and values.
- Researched and compared segmentation models, selecting standard U-Net as the starting baseline, followed by pretrained ResNet34 U-Net and DeepLabV3+, with EfficientNet-B4 considered if time and resources allow.
- Identified remaining documentation and experiment-planning tasks, including recording preprocessing decisions and defining fair model comparisons.

### Next Steps

- Define the experiment settings and evaluation protocol, keeping the geographical test split fixed.
- Complete targeted preprocessing and metric checks, and verify the environment and available hardware before modelling.
- Develop the U-Net baseline, check a forward/backward pass and test whether it can overfit one or two training pairs before full training.
