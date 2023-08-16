# Face Verification

This project focuses on implementing face verification using the ArcFace model on a dataset that predominantly contains individuals with darker skin tones. The project also utilizes a lightweight face detection model, OpenCV, to enhance processing speed.

## Objective

The primary objective of this project is to perform accurate face verification while optimizing processing time for real-world applications. The use of the ArcFace model and the choice of a lightweight face detection model contribute to achieving this objective.

## Features

- **ArcFace Model:** The project employs the ArcFace model for face recognition. ArcFace is known for its effectiveness in handling the challenges posed by varying lighting conditions and different skin tones.

- **Dataset with Dark Skin Tones:** The dataset used for training and testing emphasizes individuals with darker skin tones. This focus ensures that the model is robust and accurate across diverse skin tones.

- **OpenCV Face Detection:** To enhance the project's efficiency, a lightweight face detection model based on OpenCV is used. This choice minimizes computational resources while maintaining satisfactory accuracy in detecting faces.

## Data Preprocessing

### Face Detection

Utilized the OpenCV face detection model to detect faces in images. The choice of OpenCV enhances detection speed, contributing to real-time processing.

### Face Alignment

Applied face alignment techniques to align detected faces, standardizing facial orientations for consistent embeddings.

### Normalization

Normalized the aligned faces to ensure consistent lighting conditions and pixel values. Common techniques include histogram equalization or contrast adjustment.

## Face Embeddings Generation

Utilized the ArcFace model to generate facial embeddings from the aligned and normalized faces.

## Face Verification

- To perform face verification, compute the similarity between embeddings of input faces and the embeddings of faces present in the dataset using metrics like cosine similarity.

- Set an appropriate threshold value to determine whether the faces match or not. This threshold should be tuned based on your application's requirements and dataset characteristics. In my case, I set the threshold to 0.6.

## Requirements

Before you begin working with this project, ensure that you have the following prerequisites installed:

- **Python:** Version 3.11.4. You can download Python from the [official website](https://www.python.org/downloads/).

- Run the following command to install the dependencies:
  ```bash
  pip install -r requirements.txt
Following results are evaluated on GPU provided by Collab.

## Model Evaluation
### Speed Test
We conducted speed tests on varying sizes of datasets to evaluate the processing efficiency of our face verification system. The utilization of the lightweight OpenCV face detection model, combined with the optimized ArcFace model, resulted in impressive real-time performance without compromising accuracy. The results are evaluated on a GPU provided by Collab.

|  | 250| 500 |  3000 |  10000 |  
|-------------|--------------|--------------|--------------|--------------|
| Time taken (s)     | 0.3    | 0.5-1     |   2-2.5     |     3-4     |        


### Prediction Results
We evaluated the performance of our face verification model by testing it on a diverse range of input images. The model consistently delivered accurate predictions, successfully verifying faces across different skin tones and lighting conditions.

![Prediction Results](https://github.com/RaoSharjeelKhan/Face_registration_ArcFace/blob/main/image.png)

The following table showcases the prediction results for a sample set of input images:
| Input Image | Prediction 1 | Prediction 2 |  
|-------------|--------------|--------------|
| Nick Cannon     | Verified     | Verified     |      
| Chris Rock     | Verified     | Verified     |      
| Shaquille O'Neal       | Verified     | Verified     |      
| Samuel L. Jackson    | Verified     | Verified     |

## Conclusion 
This project has aimed to address the challenges of face verification in diverse datasets, particularly focusing on individuals with darker skin tones. By utilizing the powerful ArcFace model for face recognition and incorporating a lightweight face detection model like OpenCV, we have successfully achieved accurate and efficient face verification.




