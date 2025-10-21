# Classification of Multiple Sclerosis and Healthy Controls from EEG Signals via Poincaré Features and Deep Learning

This repository contains MATLAB scripts used in the study:

**"Poincaré Feature-Based Classification of Electroencephalography Signals for Multiple Sclerosis Diagnosis"**


🧠 Overview

The aim of this work is to classify EEG signals from individuals with Multiple Sclerosis (MS) and healthy controls using Poincaré-based nonlinear features and sub-frequency band analysis. Extracted features are then used to train and evaluate various Machine Learning (ML) and Deep Learning (DL) models.

Repository Contents


- **Main_Feature Extarction*  
  Reads EEG recordings, filters signals into frequency subbands, and computes Poincaré features (SD1, SD2, SD1/SD2 ratio, etc.).  
  The extracted feature matrix is saved in `.mat` format for later use.

- **Main_Classification**  
  Loads saved feature data and performs classification using both ML and DL models.  
  Includes standard evaluation metrics (accuracy, confusion matrix, etc.).

- **CNN_LSTM_CV*  
  Defines the **CNN+LSTM** model architecture.

- **LSTM_GRU_CV**  
  Defines the **LSTM+GRU** model architecture.



## ⚙️ Requirements

- MATLAB R2022b or later  
- Deep Learning Toolbox  
- Signal Processing Toolbox  
- Statistics and Machine Learning Toolbox  
