# StitchingNet-Seg
StitchingNet-Seg is a large-scale dataset for semantic segmentation, comprising 10,836 images with pixel-level annotations, designed to advance automated quality inspection in the textile and apparel industry.

## Motivation
Adopting Industry 4.0 technologies in the textile and apparel industry has been historically slow due to the labor-intensive nature of production and the inherent deformability of non-rigid textile materials. While automated quality inspection is well-established for raw fabric defects, sewing stitch defects, which occur dynamically during assembly, still rely heavily on manual visual checks, leading to a defect leakage rate of 5–10% and significant production bottlenecks. Existing datasets often provide only image-level labels or bounding boxes, which are insufficient for the precise geometric analysis needed for high-end quality control. To bridge this gap, we introduce StitchingNet-Seg, a large-scale semantic segmentation dataset comprising 10,836 images with meticulous pixel-level masks across 11 fabric types and 7 defect categories, providing a reliable foundation for developing intelligent Automated Optical Inspection (AOI) systems in the global garment industry.

## Dataset description
The dataset includes diverse sewing conditions and precise semantic masks to ensure model robustness.
- Total images: 10,836 images (derived and filtered from the original StitchingNet)
- Fabric varieties: 11 representative fabric types with various textures and colors
  - A. Cotton-Poly, B. Linen-Poly, C. Denim-Poly, D. Velveteen-Poly, E. Polyester-Poly, F. Satin-Core, G. Chiffon-Poly, H. Nylon-Core, I. Jacquard-Poly, J. Oxford-Core, and K. Polyester (coated)-Core
- Thread colors: Combinations of similar and contrasting thread colors.
- Classes: normal and 7 defective types
  - 0. Normal, 1. Skipped stitch, 2. Broken stitch, 3. Pinched fabric, 4. Crooked seam, 5. Thread sagging, 7. Stain and damage, and 10. Overlapped stitch
- Resolution: 224 × 224 pixels.

### Creation details
- Original source: <a href="https://github.com/hyungjungkim/StitchingNet" target="_blank">StitchingNet (14,565 sewing stitch images)</a>
- Time period (filtration and annotation): 2025.02 - 2026.01
- Annotation: Pixel-level semantic masks created using the <a href="https://cvat.ai" target="_blank">Computer Vision Annotation Tool (CVAT)</a>

### Data records
The dataset is organized into a hierarchical structure containing various fabric types and sewing defects. For the convenience of the researchers, we provided original images, segmentation masks, and annotation files in the COCO format.

### Sample images
<img src="images/stitchingnet-seg-annotation-examples.png" height="400"/>

### Code examples
We provide reference implementation codes in the [code-examples folder](/code-examples) to help researchers quickly get started with StitchingNet-Seg.

## Original publication
(TBA)

## Download data
StitchingNet data can be downloaded directly from the following repositories.
- <a href="https://doi.org/10.6084/m9.figshare.31222708" target="_blank">figshare</a>
- (under preparation) <a href="https://www.kaggle.com/datasets/hyungjung/stitchingnet-seg" target="_blank">Kaggle</a>
- (under preparation) <a href="https://data.mendeley.com/datasets" target="_blank">Mendeley data</a>

## License
The StitchingNet is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). This means it is free for research updates and non-commercial use with proper attribution.

## Contact
Please email Hyungjung Kim (hyungjungkim@konkuk.ac.kr) and Junhyeok Park(wnsgur9910@konkuk.ac.kr) with any questions regarding the dataset.

<img src="images/konkuk_university.png" height="45"/>
