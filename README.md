# ToothGapID
<!-- Repository Overview Badges -->
<div align="center">
    <img src="https://img.shields.io/github/stars/arpsn123/ToothGapID?style=for-the-badge&logo=github&logoColor=white&color=ffca28" alt="GitHub Repo Stars">
    <img src="https://img.shields.io/github/forks/arpsn123/ToothGapID?style=for-the-badge&logo=github&logoColor=white&color=00aaff" alt="GitHub Forks">
    <img src="https://img.shields.io/github/watchers/arpsn123/ToothGapID?style=for-the-badge&logo=github&logoColor=white&color=00e676" alt="GitHub Watchers">
</div>

<!-- Issue & Pull Request Badges -->
<div align="center">
    <img src="https://img.shields.io/github/issues/arpsn123/ToothGapID?style=for-the-badge&logo=github&logoColor=white&color=ea4335" alt="GitHub Issues">
    <img src="https://img.shields.io/github/issues-pr/arpsn123/ToothGapID?style=for-the-badge&logo=github&logoColor=white&color=ff9100" alt="GitHub Pull Requests">
</div>

<!-- Repository Activity & Stats Badges -->
<div align="center">
    <img src="https://img.shields.io/github/last-commit/arpsn123/ToothGapID?style=for-the-badge&logo=github&logoColor=white&color=673ab7" alt="GitHub Last Commit">
    <img src="https://img.shields.io/github/contributors/arpsn123/ToothGapID?style=for-the-badge&logo=github&logoColor=white&color=388e3c" alt="GitHub Contributors">
    <img src="https://img.shields.io/github/repo-size/arpsn123/ToothGapID?style=for-the-badge&logo=github&logoColor=white&color=303f9f" alt="GitHub Repo Size">
</div>

<!-- Language & Code Style Badges -->
<div align="center">
    <img src="https://img.shields.io/github/languages/count/arpsn123/ToothGapID?style=for-the-badge&logo=github&logoColor=white&color=607d8b" alt="GitHub Language Count">
    <img src="https://img.shields.io/github/languages/top/arpsn123/ToothGapID?style=for-the-badge&logo=github&logoColor=white&color=4caf50" alt="GitHub Top Language">
</div>

<!-- Maintenance Status Badge -->
<div align="center">
    <img src="https://img.shields.io/badge/Maintenance-%20Active-brightgreen?style=for-the-badge&logo=github&logoColor=white" alt="Maintenance Status">
</div>


ToothGapID is an innovative tool designed for the precise detection of missing teeth in dental X-rays, utilizing state-of-the-art deep learning models to facilitate reliable and standardized reporting based on the **FDI numbering system**. This project integrates advanced image processing techniques to enhance diagnostic capabilities and improve patient outcomes.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Model Performance](#model-performance)
  - [Detectron2 Performance](#detectron2-performance)
  - [YOLOv8 Performance](#yolov8-performance)
- [Challenges and Failures](#challenges-and-failures)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Features

- **FDI-Based Detection**: Provides accurate identification of missing teeth using the FDI numbering system, ensuring standardized communication among dental professionals.
- **High Detection Accuracy**: Leveraging deep learning, ToothGapID achieves impressive detection rates across diverse datasets.
- **Flexible Integration**: Designed for easy adaptation into various dental imaging workflows, allowing for seamless integration into existing clinical systems.
- **Extensive Documentation**: Comprehensive guides for setup, usage, and customization ensure accessibility for users with varying levels of technical expertise.

## Installation

To set up ToothGapID, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/arpsn123/ToothGapID.git
   cd ToothGapID
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```


## Technologies Used

This project leverages a variety of powerful tools and libraries to achieve its objectives. Below is a list of the key technologies utilized in ToothGapID, along with their respective badges:

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6.0-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0-red.svg) ![OpenCV](https://img.shields.io/badge/OpenCV-4.5.3-blue.svg) ![Detectron2](https://img.shields.io/badge/Detectron2-0.5.1-orange.svg) ![YOLOv8](https://img.shields.io/badge/YOLOv8-latest-yellowgreen.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21.0-orange.svg) ![Pandas](https://img.shields.io/badge/Pandas-1.3.0-green.svg) ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4.2-blue.svg) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24.2-yellow.svg) 
![Keras](https://img.shields.io/badge/Keras-2.6.0-red.svg) ![Jupyter](https://img.shields.io/badge/Jupyter-1.0.0-orange.svg) ![VSCode](https://img.shields.io/badge/VSCode-1.58.0-blue.svg) ![Git](https://img.shields.io/badge/Git-2.32.0-orange.svg)  ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13.3-blue.svg) 
![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU%20Support-76B900.svg) ![CUDA](https://img.shields.io/badge/CUDA-11.2-76B900.svg) ![cuDNN](https://img.shields.io/badge/cuDNN-8.1.0-76B900.svg) ![OpenAI](https://img.shields.io/badge/OpenAI-API-76B900.svg) ![Pillow](https://img.shields.io/badge/Pillow-8.2.0-red.svg) ![Requests](https://img.shields.io/badge/Requests-2.25.1-blue.svg)

## Model Performance

ToothGapID leveraged two prominent deep learning models—**Detectron2** and **YOLOv8**—to evaluate their effectiveness in detecting missing teeth.

### Detectron2 Performance
- **Overview**: Initially, Detectron2 was chosen for its robust segmentation capabilities and adaptability to various object detection tasks. However, its performance did not meet expectations due to several critical factors.

- **Weaknesses**:
  - **Annotation Issues**: 
    - The performance was significantly hindered by the presence of incorrect annotations within the training dataset. 
    - Mislabeling led to poor model training, resulting in low detection accuracy and significant misclassifications.
  - **Generalization Failures**:
    - The model struggled to generalize beyond the training data, showing weak performance on unseen images, thereby limiting its practical application in real-world scenarios.
  
#### Visual Representation of Detectron2 Performance

| Metric                | Value                |
|-----------------------|---------------------|
| **Precision**         | Low due to errors    |
| **Recall**            | Inconsistent         |
| **mAP@0.5**          | Below expectations   |

### YOLOv8 Performance
- **Overview**: YOLOv8 was implemented due to its high efficiency and speed in real-time object detection, proving to be a more suitable choice for this project.

- **Strengths**:
  - **High Detection Rates**:
    - The model excelled in accurately identifying missing teeth, aligning with the project's objective of flagging absent teeth effectively.
  - **Recognition without Segmentation**:
    - Notably, while YOLOv8 successfully identified missing teeth, it did not segment the exact locations of the missing teeth. This limitation underscored a crucial aspect of the project's goals—visual representation of missing teeth regions—which was not achieved despite successful detection.

#### Visual Representation of YOLOv8 Performance

| Metric                | Value                |
|-----------------------|---------------------|
| **Precision**         | High (85%+)         |
| **Recall**            | Moderate (70%)      |
| **mAP@0.5**          | Good (75%)          |

### Summary of Model Comparison

| Model      | Detection Accuracy | Segmentation Capability | Annotation Quality Impact |
|------------|--------------------|-------------------------|---------------------------|
| Detectron2 | Poor               | Yes                     | High                      |
| YOLOv8     | Good               | No                      | Moderate                  |

## Challenges and Failures

Throughout the development of ToothGapID, several challenges were encountered:

1. **Data Variability**: 
   - The diversity in X-ray imaging conditions—including varying resolutions and noise levels—resulted in inconsistent model performance, affecting detection accuracy.

2. **Annotation Quality**:
   - Detectron2's performance failures highlighted the necessity for high-quality, precise annotations. Incorrect labels in the training set were a significant contributor to the model's inadequacy.

3. **Class Imbalance**:
   - The dataset's imbalance, with a significantly higher number of images representing non-missing teeth, biased the models toward over-representing these classes. This imbalance was addressed through data augmentation and the implementation of weighted loss functions to improve the learning process.

4. **Integration and Adaptation**:
   - Adapting ToothGapID for clinical use required extensive testing to ensure compatibility with existing systems, as well as modifications to output formats to align with the requirements of dental professionals.

## Future Work

To enhance ToothGapID's capabilities, several future directions are proposed:

- **Refinement of Annotations**: 
  - A concerted effort to rectify annotation inaccuracies is essential for improving model performance, particularly for Detectron2.

- **Model Fine-Tuning**: 
  - Further fine-tuning of YOLOv8 to include segmentation capabilities for visual representation of missing teeth locations, addressing one of the critical limitations observed during initial testing.

- **Expanding Dataset**: 
  - Augmenting the dataset with additional labeled images to better represent various conditions and improve generalization across diverse dental imaging scenarios.

- **Clinical Validation**: 
  - Conducting clinical trials to validate the effectiveness and reliability of ToothGapID in real-world settings, ensuring that the tool meets the needs of dental practitioners.

## Contributing

Contributions are welcome! If you have suggestions for improvements or want to report issues, please create an issue or submit a pull request. For larger contributions, please consider discussing them in an issue first to ensure alignment with project goals.


## Acknowledgments

- Gratitude to dental professionals who provided insights and feedback during the testing phases, enhancing the model's applicability in real-world scenarios.
- Appreciation to the developers and researchers behind Detectron2 and YOLOv8 for their groundbreaking work in deep learning and computer vision, which formed the foundation of this project.

