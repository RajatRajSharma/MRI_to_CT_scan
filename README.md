# Data set:-
https://zenodo.org/records/7260705 

# Models :-
https://drive.google.com/drive/u/1/folders/1mtTygiVIjpb8gmSn3h2RKz272qI7S2-i

# Best Models code Flow :-

### MK3.1.2_GOD_MRI_CT.ipynb ( 17/01/2025 )
- MSE: 0.063136, SSIM: 0.449328, FID: 221.746994, VGG Loss: 2.912386

### MK3.1.2_v4_50epochs_GOD.ipynb ( 24/04/2025 )
- MSE: 0.072253, SSIM: 0.702594, FID: 174.542770, VGG Loss: 2.671596

## Check Point 1 (26/05/2025)

### MK6.0.1_CP1
- MSE: 0.065860, SSIM: 0.678714, FID: 175.659180, VGG Loss: 2.517590

### MK6.0.3_CP1
- MSE: 0.056884, SSIM: 0.591850, FID: 166.765396, VGG Loss: 2.724192

### MK6.1.0_CP1 (30/05/2025)
- MSE: 0.019255, SSIM: 0.825634, FID: 113.451118, VGG Loss: 1.631517

### MK6.1.1_CP1 (31/05/2025)
- MSE: 0.019586, SSIM: 0.813731, FID: 118.909042, VGG Loss: 1.640784

## Check Point 2 (01/06/2025)

### MK6.2.0_CP2
- MSE: 0.018466, SSIM: 0.823152, FID: 113.933151, VGG Loss: 1.602489

### MK6.2.1_CP2
- MSE: 0.015199, SSIM: 0.852237, FID: 104.621910, VGG Loss: 1.428501

## Check Point 3 (05/06/2025)

## MK6.3.0_CP3
- MSE: 0.013473, SSIM: 0.866290, FID: 97.263969, VGG Loss: 1.324233

### 

# Ideas to think about
- Done: 2 type of data Augmentation required
- Done: Will have updated tranformation function that will work on 2 images in pair like this
```
mri_image, ct_image = self.transform(mri_image, ct_image)
```
-  Done: 3 channel to 1 channel as MRI/CT is BW only
-  Done: Will not use large train data (atmost 1000 pair) if all the data doesn't handle lot of cases.
- Done: Will have 3 center data
  
# Metric Selection for MRI-to-CT Generation in HMSS

## Low-Resolution Generator (LRG):
- **MSE/MAE**: Useful for pixel-level accuracy in low-resolution images where fine details aren't critical.
- **SSIM**: Assesses perceptual similarity, focusing on structure and texture, which is important in low-res images.
- **PSNR**: Measures overall image fidelity, providing a quantitative view of quality in low-resolution images.

## High-Resolution Generator (HRG):
- **PSNR**: Evaluates peak signal-to-noise ratio in high-resolution images, which is important for preserving detail.
- **SSIM**: Ensures perceptual similarity, especially in high-resolution images where structural integrity is key.
- **Perceptual Loss**: Aligns high-level features, preserving semantic content in high-resolution images.
- **FID**: Measures the statistical similarity between the real and generated image distributions, ensuring realism.
- **Dice Similarity Coefficient (Bone_DSC)**: Evaluates the segmentation accuracy of bone structures, important for medical images.

## Why Not Use MSE in HR?
- **MSE** is sensitive to pixel-level differences, which may not correlate with perceptual quality. In high-resolution medical images, where fine details and structural integrity are critical, metrics like **SSIM** and **Perceptual Loss** are more meaningful as they better reflect the perceptual quality and structure of the image.

## Why Choose SSIM for LR Instead of MSE?
- **SSIM** considers luminance, contrast, and structure differences, making it more robust for evaluating perceptual quality in low-resolution images. Unlike **MSE**, it better reflects human visual perception, which is critical when dealing with low-res images where fine details may be lost.
