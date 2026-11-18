# Land Cover Mapping Using Semantic Segmentation

Fontys ICT Semester 3 — Main Challenge.

**Research question:** How accurately can CNN-based semantic segmentation models classify land-cover types, and how well do they generalize to unseen geographical regions?

## Project Overview

This project investigates how deep learning can identify land-cover types in aerial and satellite images. Using the OpenEarthMap dataset, it focuses on semantic segmentation: assigning a class, such as building, road, tree or water, to each pixel in an image.

## Problem

Manually mapping land cover is time-consuming. Automatic classification is also challenging because different surfaces can look similar, while landscapes, lighting and image resolution vary between locations. A model that performs well on familiar regions may struggle in a new geographical area.

## Proposed Solution

The project aims to train and compare CNN-based segmentation models, starting with U-Net, and evaluate their predictions using overall and per-class metrics. Separate geographical regions will be used for training, validation and testing to assess performance on unseen locations.

A simple application is planned to let users upload an aerial image and view its predicted land-cover mask and class percentages.

## Why It Matters

Land-cover information can support environmental research and urban planning, for example by helping identify the distribution of vegetation and built surfaces. This project explores the potential and limitations of automating that analysis, with particular attention to reliability across geographical regions. Its predictions are estimates from an academic prototype, rather than validated planning evidence.
