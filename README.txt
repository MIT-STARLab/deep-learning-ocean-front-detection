This project was creating ML models to detect ocean fronts using Landsat imagery; described in detail in the thesis. I downloaded relevant Landsat oceanic images using Google Earth Engine, created training/testing/ground truth datasets using Google Drive, and trained ML models on these datasets using MIT SuperCloud.

Below is a brief guide to what code is where:

google_drive (this code was run on Google Drive)
	atlantic_viz - visualizations of all training data from Atlantic Ocean
	Augmenting_Data - augmenting, thresholding, and scaling training data
	Creating_Ground_Truth - applying BOA to Landast bands to create training data
	human_annotated_viz - visualizations of all human annotated training data
	indian_viz - visualizations of all training data from Indian Ocean
	Model_Runtimes - computing model runtimes using various hardware
	pacific_viz - visualizations of all training data from Pacific Ocean

google_earth_engine (this code was run on Google Earth Engine)
	Data_Pipeline - downloading Landsat band training input data

mit_supercloud (this code was run on MIT SuperCloud)
	fine_tune - fine tuning BOA-trained models on human annotated data
	metrics - metrics for all trained models
	models - architectures for all models
	Plotting_Metrics - plotting prc and calculating binary metrics for all trained models
	Plotting_Predictions - plotting sample inputs/ground truths/predictions for all trained models
	Plotting_SSIM - plotting input/prediction SSIM for all trained models
	test - calculating metrics for all trained models
	train - training all models

The training datasets and trained models were too big to upload to GitHub, so they can be found here:
https://drive.google.com/drive/folders/1wzsIlGT01_KCKoa9FyUhcCaREzG3n0p_?usp=sharing

Input data is of shape 224x224x3 (the 3 channels are Landsat bands 1, 3, and 10) and output data is of shape 224x224x2 (the 2 channels are CHL fronts and SST fronts)