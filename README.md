# Data-Driven Discoveries in Pet Adoption Patterns

## Project Overview
This project aims to uncover factors influencing animal adoption speeds by analyzing a comprehensive dataset of adopted pet profiles. By identifying hidden patterns that impact adoption speed, we hope to help shelters create more appealing online profiles, ultimately improving adoption rates and benefiting global animal welfare.

## Team Members
- Yushan Shi (ys468)
- Yuqi Su (ys457)
- Yurui Wu (yw695)
- Mingxin Liu (ml754)

## Dataset
We're using the PetFinder.my Adoption Prediction dataset from Kaggle. This dataset contains detailed information on pet profiles, including:
- Pet characteristics
- Health conditions
- Adoption speed

## Methodology
Our analysis involves the following steps:

1. **Exploratory Data Analysis (EDA)**
   - Visualize distributions and relationships
   - Conduct statistical tests (e.g., ANOVA)

2. **Feature Importance Analysis**
   - Train ensemble models (e.g., Random Forest)
   - Use LIME/SHAP values for detailed explanations

3. **Advanced Visualizations**
   - Create Partial Dependence Plots
   - Visualize decision trees

4. **Insight Extraction**
   - Identify surprising patterns in pet adoption
   - Explain causality between pet features and adoption speed

## Project Structure
- `load_data.py`: Contains functions for loading and preprocessing data
- `main.py`: Main script for running the analysis pipeline
- `model.py`: Defines the XGBoost model class
- `tuning.py`: Functions for hyperparameter tuning
- `feature_importance.py`: Implements SHAP analysis for feature importance
- `config.py`: Contains configuration parameters

## How to Run
1. Ensure you have all required dependencies installed.
2. To run the analysis with default settings:
 python main.py
3. To run with hyperparameter tuning:
 python main.py --tune
4. To include sentiment features in the analysis:
 python main.py --sentiment


## Evaluation
We use both quantitative and qualitative measures to evaluate our findings:

**Quantitative Measures:**
- F1 score
- Accuracy
- AUC-ROC Curve
- ANOVA/Chi-Square Tests
- Correlation Analysis

**Qualitative Evaluation:**
- Random Forest and SHAP/LIME value interpretations
- Partial dependence plot visualizations

## Goals
While our primary goal is to predict adoption speed, we're more focused on understanding the factors that influence the adoption process. We aim to provide actionable insights that can help increase adoption rates and improve animal welfare.
