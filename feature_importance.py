# feature_importance.py
import shap
import matplotlib.pyplot as plt
import numpy as np

def calculate_shap_values(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    return shap_values

def plot_shap_summary(shap_values, X, output_file=None):
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    
    if output_file:
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()

def plot_shap_dependence(shap_values, X, feature_names, output_dir=None):

    for i in range(min(10, len(feature_names))):
        shap.dependence_plot(feature_names[i], shap_values, X, show=False)
        
        if output_dir:
            plt.savefig(f"{output_dir}/dependence_plot_{feature_names[i]}.png")
            plt.close()
        else:
            plt.show()

def analyze_feature_importance(model, X, feature_names, output_dir=None):
    shap_values = calculate_shap_values(model, X)
    
    plot_shap_summary(shap_values, X, 
                      output_file=f"{output_dir}/shap_summary.png" if output_dir else None)

    plot_shap_dependence(shap_values, X, feature_names, output_dir)

    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = dict(zip(feature_names, mean_shap))
    return feature_importance