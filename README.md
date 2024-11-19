# Multimodal Deep Learning for the Prediction of Crop Phenotype using Heterogenuous data

Master Thesis Project

## Project Description

This master thesis presents a novel multimodal deep learning approach for crop phenotype prediction by integrating multiple data modalities through an intermediate fusion architecture. The project focuses on predicting barley's flowering time and yield using genetic markers, environmental data, and study conditions.

## Research Objectives
- Integrating different modalities of data (genetic data, environment data, study conditions) to enhance the prediction accuracy of the model.
- Development of fusion strategies to combining CNN and LSTM networks 
- Integration of high-dimensional genetic data (30,543 markers) with environmental time series
- Evaluation of model performance against traditional single-modality approaches
- Contribution to precision agriculture through improved phenotype prediction

## Model Architecture

### Intermediate Fusion Model
![Intermediate Fusion Architecture](images/intermediate_fusion.png)

The intermediate fusion model combines environmental, genetic, and study data early in the network architecture.
4-layer CNN was used to extract high latent features from the Genetic markers (30,543 features), and 2-stack LSTM was to used extract high latent features from environmental data. Additionally embedding was used to process categorical study conditions. 

### Late Fusion Model
![Late Fusion Architecture (baseline model)](images/late_fusion.png)

Random Forest is used as one of the model to precess gene data in this fusion strategy as Random Forest has shown higher results.

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Overview

### Genetic Data
- Sourced from Western Barley Genetic Alliance (WCGA), Murdoch University
- 894 barley accessions genotyped using Next-Generation Sequencing
- 30,543 SNP markers after quality filtering:
 - Heterozygosity filtering
 - Mapping quality threshold: 20
 - Major allele frequency (MAF): 0.01
 - Average marker density: ~150kb

### Field Trial Data
- Locations: Geraldton, Merredin, South Perth, Katanning, and Esperance (Western Australia)
- Timeline: 12 experiments (2015-2016)
- Special Trials:
 - Light Exposure: South Perth (2016)
   - 18-hour artificial lighting vs. natural conditions
 - Irrigation: Merredin
   - Irrigated vs. non-irrigated conditions
Each trial site represents distinct climatic zones, providing diverse environmental conditions for phenotype assessment.

### Environmental Data
- Source: Australian Bureau of Meteorology (BOM)
- Data collected for each trial location:
 - Rainfall
 - Temperature
 - Solar radiation
- Time period: March to July (5 months)
- Locations:
 - Geraldton
 - Merredin
 - South Perth
 - Katanning
 - Esperance

 Environmental data corresponds to the specific geographic locations where field trials were conducted, enabling direct correlation between weather conditions and crop performance.

 




