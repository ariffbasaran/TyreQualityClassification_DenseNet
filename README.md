# Tyre Quality Classification with DenseNet

This project aims to classify tire images into two categories: "Good" and "Defective" using a DenseNet model. The dataset for this project was obtained from Kaggle.

## Model Evaluation Results

The model has been evaluated on the test dataset, and the following results were obtained:

- **Test Set Accuracy (Mean):** 95.43%
- **Recall:** 0.945783
- **Precision:** 0.951515
- **Confusion Matrix:**
  
![image](https://github.com/ariffbasaran/TyreQualityClassification_DenseNet/assets/109107707/408fc83d-87c0-4741-a753-13b73e65e08e)

## About Dataset

The dataset contains 1854 digital tyre images, categorized into two classes: defective and good condition. Each image is in a digital format and represents a single tyre. The images are labelled based on their condition, i.e., whether the tyre is defective or in good condition.

This dataset can be used for various machine learning and computer vision applications, such as image classification and object detection. Researchers and practitioners in transportation, the automotive industry, and quality control can use this dataset to train and test their models to identify the condition of tyres from digital images. The dataset provides a valuable resource to develop and evaluate the performance of algorithms for the automatic detection of defective tyres.

The dataset may also help improve the tyre industry's quality control process and reduce the chances of accidents due to faulty tyres. The availability of this dataset can facilitate the development of more accurate and efficient inspection systems for tyre production.

[Dataset Link](https://www.kaggle.com/datasets/warcoder/tyre-quality-classification/data)

## Requirements

The following libraries and tools have been used for the development and execution of the project:

- [NumPy](https://numpy.org/): A Python library used for numerical computations.
- [Pandas](https://pandas.pydata.org/): A Python library for data analysis and manipulation.
- [PyTorch](https://pytorch.org/): An open-source deep learning framework used for machine learning and deep learning.
- [TorchVision](https://pytorch.org/vision/stable/index.html): A library for image processing and modeling with PyTorch.
- [Matplotlib](https://matplotlib.org/): A Python library used for data visualization.
- [scikit-learn](https://scikit-learn.org/stable/): A Python library used for machine learning.
- [OpenCV](https://opencv.org/): A library used for image processing and computer vision.
- [PIL (Pillow)](https://pillow.readthedocs.io/en/stable/): An image processing library for Python.

## Installation

To run the project in your local environment, you may need to install the required libraries first. You can install them using the following commands:

```bash
pip install numpy pandas torch torchvision matplotlib scikit-learn opencv-python pillow
