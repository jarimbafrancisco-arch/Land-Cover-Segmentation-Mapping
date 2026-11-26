# Notebook 05: pretrained U-Net with ResNet34

## Purpose

Compare a pretrained U-Net with ResNet34 against the lightweight U-Net in
`04_baseline_experiment.ipynb`, using the same training and validation protocol.
The hypothesis is improved validation segmentation quality; no improvement is
assumed and the test regions remain reserved for final evaluation.

## Approaches and selected configuration

- ResNet34 with random initialization would compare training from scratch.
- ResNet34 initialized with ImageNet weights adds transfer learning. The student
  selected this option. All encoder and decoder parameters remain trainable.
- Freezing the pretrained encoder could reduce backward computation, but is not
  the selected experiment.

Use `segmentation_models_pytorch.Unet` with `encoder_name="resnet34"`,
`encoder_weights="imagenet"`, three input channels, eight output channels and
`activation=None`. Use the library's U-Net decoder, documenting its configuration
and parameter count. This changes model capacity and decoder implementation as
well as initialization; results cannot isolate the effect of pretraining alone.

## Constants verified in notebook 04

Keep the saved geographical split manifest, 512 x 512 resizing, batch size 8,
30 epochs, seed 42, Adam with learning rate 0.001, num_workers=0, shuffled training
and deterministic validation loading. Keep weighted Cross-Entropy, inverse
square-root training pixel frequencies normalized to mean 1, and ignore_index=255.
Retain global confusion-matrix accumulation, IoU/Dice absent-class handling and
best checkpoint selection by validation mIoU. Do not add augmentation, a scheduler,
early stopping, AMP or gradient accumulation only to notebook 05.

## Input normalization

Proposed additional input operation: use the normalization specified by the
ImageNet encoder weights on the RGB tensors already scaled to [0, 1]. Apply it
consistently at the model input for forward checks, training, validation and
subsequent inference. Keep the shared dataset implementation and masks unchanged;
visualizations continue using the original [0, 1] tensors. Store normalization
parameters with the checkpoint. This documented input change needs approval under
the project preprocessing rule.

## Deliverables

Create `notebooks/05_unet_resnet34.ipynb`, following the readable section order of
notebook 04 and explaining the encoder, decoder, transfer learning and normalization.
Add the required dependency to `requirements.txt` after checking version compatibility.
Use distinct ResNet34 checkpoint, history and per-class result filenames so baseline
artifacts cannot be overwritten. Record model settings, weights identifier, library
versions, seed, split identity, class weights, normalization, timing and parameter
count. Do not modify notebook 04 or regenerate its splits.

## Verification and execution boundaries

Check notebook syntax, equivalent protocol settings and the metric calculations.
Once dependencies and pretrained weights are available, verify output shape
`(1, 8, 512, 512)`, finite logits/loss/gradients and that all parameters are trainable.
Verify a full batch of 8 fits on the intended training device before full training;
do not silently lower the batch or resolution if it fails. The target GPU and its
memory have not been verified. Passing single-image checks does not prove batch 8 fits.

Installation, pretrained-weight download and training are separate execution
steps. Preparing the notebook does not initiate them. Full training remains gated
on the project's preprocessing, metrics and tiny-overfit checks.

## Sources

- https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/docs/quickstart.rst
- https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/docs/models.rst
- https://huggingface.co/smp-hub/resnet34.imagenet

Status: selected model and comparison protocol recorded; awaiting approval of this
design, including the proposed ImageNet input normalization, before implementation.
