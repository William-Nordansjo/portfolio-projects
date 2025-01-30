#This python script will contain supporting functions to the the breast cancer classification project
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.base import BaseEstimator


#Function 1: Creating ML models
def fit_and_predict_models(models: dict, X_train, y_train, X_test):
    predictions = {}
    
    for name, model in models.items():
        if isinstance(model, BaseEstimator):  # Valid model?
            model.fit(X_train, y_train)  
            predictions[name] = model.predict(X_test)  
            
    return predictions

#Function 2: Creating ML models
def fit_and_predict_probabilites_models(models: dict, X_train, y_train, X_test):
    predicted_probabilities = {}
    
    for name, model in models.items():
        if isinstance(model, BaseEstimator):  # Valid model?
            model.fit(X_train, y_train)    
            predicted_probabilities[name] = model.predict_proba(X_test)[:, 1]
            
    return predicted_probabilities

#Function 3: Creating multiple confusion matrices and plots them
def plot_confusion_matrices(confusion_matrices, titles, figsize=(18, 12)):
    # Creating a 2x3 subplot structure
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Plotting each confusion matrix
    for ax, cm, title in zip(axes.flatten(), confusion_matrices, titles):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_title(f'{title} Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Hiding the sixth subplot (bottom right)
    if len(confusion_matrices) < 6:
        fig.delaxes(axes[1, 2])
    
    # Adjusting layout
    plt.tight_layout()
    plt.show()
    
    
#Function 4: Performance metrics
def calculate_metrics(y_test, predictions, predicted_probabilities):
    metrics = {'Model': [], 'Accuracy': [], 'AUC': []}
    
    for name, y_pred in predictions.items():
        y_pred_prob = predicted_probabilities.get(name)
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else None
        
        metrics['Model'].append(name)
        metrics['Accuracy'].append(round(accuracy, 3))
        metrics['AUC'].append(round(auc, 3) if auc is not None else 'N/A')
    
    # Convert to DataFrame for better display
    metrics_df = pd.DataFrame(metrics)
    return metrics_df
