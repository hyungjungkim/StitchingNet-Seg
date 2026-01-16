# StitchingNet-Seg
StitchingNet-Seg is a large-scale dataset featuring 10,836 images with pixel-level annotations, designed to advance automated quality inspection in the textile and apparel industry.

## Motivation
Automated quality control in the garment industry is challenging due to the labor-intensive tasks and non-rigid nature of textiles. While previous research focused on fabric-level defects (e.g., holes or stains), sewing stitch defects, which occur during the fabric joining process, have lacked high-quality, pixel-level datasets. StitchingNet-Seg bridges this gap by providing precise semantic segmentation masks. Unlike simple labels and bounding boxes, these masks capture intricate geometric features such as defect shape, size, and orientation, enabling AI models to perform root-cause analysis and real-time process monitoring for smart manufacturing applications.

## Dataset description
The dataset includes diverse sewing conditions to ensure model robustness:
- Total images: 10,836 images (derived and filterred from the original StitchingNet).
- Fabric varieties: 11 representative fabric types with various textures and colors
- Thread colors: Combinations of similar and contrasting thread colors.
- Classes: normal and 7 defective types
  - Broken stitch, 
- Resolution: 224 × 224 pixels.

### Creation details
- Original source: [StitchingNet(14,565 sewing stitch images)](https://github.com/hyungjungkim/StitchingNet)
- Time period (filteration and annotation): 2025.00 - 2025.00
- Annotation: Pixel-level semantic masks created using [Computer Vision Annotation Tool (CVAT)](https://cvat.ai)

### Sample images (total 00 images)
<img src="images/stitchingnet-seg-annotation-examples.png" height="400"/>

## Original publication
(TBA)

## Download data
StitchingNet data can be downloaded directly from the following repositories:
- [Kaggle](https://www.kaggle.com/datasets/hyungjung/stitchingnet-seg)
- [Mendeley data](https://data.mendeley.com/datasets)

## License
The StitchingNet is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). This means it is free for updated research and non-commercial use with proper attribution.

## Contact
Please email Hyungjung Kim (hyungjungkim@konkuk.ac.kr) and Junhyeok Park(wnsgur9910@konkuk.ac.kr) for any questions regarding the dataset.

<img src="images/konkuk_university.png" height="45"/>
