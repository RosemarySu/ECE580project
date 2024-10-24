# feature_importance.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

def plot_feature_importance(importance_dict, output_file=None):
    plt.figure(figsize=(10, 6))
    features = list(importance_dict.keys())
    importance = list(importance_dict.values())
    
    plt.bar(range(len(importance)), importance)
    plt.xticks(range(len(features)), features, rotation=45, ha='right')
    plt.title('Feature Importance')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, output_file=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if output_file:
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()
    
    return cm

def analyze_feature_importance(model, X, feature_names, output_dir=None):
    """Analyze feature importance and model performance"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get feature importance from the model
    importance = model.feature_importances_
    
    # Create and sort feature importance dictionary
    feature_importance = dict(zip(feature_names, importance))
    sorted_importance = dict(sorted(feature_importance.items(), 
                                  key=lambda x: x[1], 
                                  reverse=True))
    
    # Plot feature importance
    if output_dir:
        plot_feature_importance(sorted_importance, 
                              f"{output_dir}/feature_importance.png")
    
    # Get predictions for confusion matrix
    y_pred = model.predict(X)
    
    # Create confusion matrix
    if output_dir:
        cm = plot_confusion_matrix(model.predict(X), y_pred, 
                                 f"{output_dir}/confusion_matrix.png")
    
    return sorted_importance