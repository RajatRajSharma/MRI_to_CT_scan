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
