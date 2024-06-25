# Sounify-Accent-Classification

**About:** 
Sounify is an AI prototype deployed on Streamlit as a web-based Python application, trained using a combination models of deep learning (CNN) and machine learning (KNN, Random Forest, and Decision Tree). Sounify can detect 13 different accents from around the world, using data sourced from the Kaggle platform: https://www.kaggle.com/datasets/rtatman/speech-accent-archive

**Pre-processing**
Before training the models, there are essential steps are required for pre-processing starting from feature extraction with MFCC, data labeling, data oversampling, label encoding, and splitting the data for training and testing purposes with an 80:20 ratio. Since the dataset has an unbalanced amount of data for each class, therefore we decided to select only 13 classes that have more than 30 audio samples from the original dataset.

**Scenarios:**
For the machine learning, each model is divided into 2 scenarios: (1) trained with a 10-seconds audio sample and (2) trained with a 5-seconds audio sample. Whereas the deep learning are also divided into 2 scenarios, (1) with a standard values and (2) employing hyperparameter tuning aided by the random search feature from Keras Tuner library.

**Comparison:**
| ML Models      | 10 seconds | 5 seconds  |
| -------------- | ---------- | ---------- |
| KNN            | 0.89       | 0.90       |
| Random Forest  | 0.99       | 0.99       |
| Decision Tree  | 0.95       | 0.96       |

| DL Models      | Without Tuning | Tuning  |
| -------------- | -------------- | ------- |
| CNN            | 0.96           | 0.98    |

Based on the experiment above, we can conclude the neural networks are not always be the best or most accurate option, as the ML models also show good performance too. Furthermore, a 98% accuracy in CNN models cannot be conclusively deemed the best performance, since the loss and accuracy graphs indicates an overfitting.

![Speech Recognition - Sounify](https://github.com/verrenangelinasaputra/Sounify-Accent-Classification/assets/131938319/4a460546-a79a-42cb-a32f-474affbeadef)
