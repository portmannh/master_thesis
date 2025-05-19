# Brain Age Prediction from Sleep EEG Data using Foundation Model Principles
## Hannah Portmann's Master Thesis

## Overview
This repository contains all code implemented by Hannah Portmann for the Master thesis on brain age prediction from sleep EEG data. The approach uses foundation model principles by extracting features via a pretext task and using those for the downstream task of brain age prediction. The datasets that were used are not included.

## Methodology
- Demographic description of the datasets
- Preparation of the datasets for automated sleep staging
- Implementation, training and evaluation of a pretext task doing automated sleep staging
- Extraction of features from the pretext task
- Training and evaluating different models for brain age prediction
- Training and evaluating different models for MCI classification

## Repository Structure
- '/datasets': scripts for statistical descriptions of datasets
- '/correlational_analyses': scripts for correlating sleep stage percentages and durations with age and plotting the correlations
- '/pretext_task': scripts for training and testing the automated sleep staging model, as well as computing different evaluation metrics and plotting results
- '/data_preparation': scripts for preparing datasets to be used as input for pretext task
- '/age_regression': scripts for extracting features from the pretext task model and training and evaluating different age regression models and plotting the results
- '/mci_classification': scripts for training and evaluating models for MCI classification and plotting the results
- 'my_usleep.py': implementation of automated sleep scoring model for pretext task (based on Perslev et al., 2021)
- 'requirements.txt': requirements for environment to run these scripts

## References
Perslev, M., Darkner, S., Kempfner, L., Nikolic, M., Jennum, P. J., & Igel, C. (2021). U-Sleep: Resilient high-frequency sleep staging. Npj Digital Medicine, 4(1), 72. https://doi.org/10.1038/s41746-021-00440-5
