import sklearn.metrics as skl_metric
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay

def matrix_summary(y_test, y_pred_model1, y_pred_model2, model1, model2):
    """
    This function creates confusion matrices and classification reports 
    for two models and displays them side by side with inverted orientation.
    
    Parameters:
    y_test : array-like
        True labels.
    y_pred_model1 : array-like
        Predictions for the first model.
    y_pred_model2 : array-like
        Predictions for the second model.
    model1 : str
        Name/label for the first model.
    model2 : str
        Name/label for the second model.
    """
    # Confusion matrices for both models (invert rows and columns for desired layout)
    conf_matrix1 = skl_metric.confusion_matrix(y_test, y_pred_model1)[::-1, ::-1]
    conf_matrix2 = skl_metric.confusion_matrix(y_test, y_pred_model2)[::-1, ::-1]

    # Classification reports for both models
    report1 = skl_metric.classification_report(y_test, y_pred_model1)
    report2 = skl_metric.classification_report(y_test, y_pred_model2)

    # Create the figure with subplots
    fig, ax = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [1, 2]})

    # Model 1 confusion matrix
    sns.heatmap(conf_matrix1, annot=True, cmap='Oranges', fmt='g', cbar=False, 
                xticklabels=["Fraud", "Not Fraud"], yticklabels=["Fraud", "Not Fraud"], ax=ax[0, 0])
    ax[0, 0].set_title(f"Confusion Matrix - {model1}")
    ax[0, 0].set_xlabel("Prediction")
    ax[0, 0].set_ylabel("True Value")

    # Model 2 confusion matrix
    sns.heatmap(conf_matrix2, annot=True, cmap='Blues', fmt='g', cbar=False, 
                xticklabels=["Fraud", "Not Fraud"], yticklabels=["Fraud", "Not Fraud"], ax=ax[1, 0])
    ax[1, 0].set_title(f"Confusion Matrix - {model2}")
    ax[1, 0].set_xlabel("Prediction")
    ax[1, 0].set_ylabel("True Value")

    # Classification report for Model 1
    ax[0, 1].text(0, 0.5, report1, fontsize=12, family='monospace', va='center', ha='left')
    ax[0, 1].set_axis_off()
    ax[0, 1].set_title(f"Classification Report - {model1}")

    # Classification report for Model 2
    ax[1, 1].text(0, 0.5, report2, fontsize=12, family='monospace', va='center', ha='left')
    ax[1, 1].set_axis_off()
    ax[1, 1].set_title(f"Classification Report - {model2}")

    # Adjust layout
    plt.tight_layout()
    plt.show()

def PRD(y_test, y_pred_model1, y_pred_model2, label_model1, label_model2):
    """
    Plots a Precision-Recall curve for two models on the same graph.
    
    Parameters:
    y_test : array-like
        True labels.
    y_pred_model1 : array-like
        Predicted probabilities or scores for the first model.
    y_pred_model2 : array-like
        Predicted probabilities or scores for the second model.
    label_model1 : str, optional
        Label for the first model (default: "Model 1").
    label_model2 : str, optional
        Label for the second model (default: "Model 2").
    """
    # Plot Precision-Recall for Model 1
    disp1 = PrecisionRecallDisplay.from_predictions(
        y_test, y_pred_model1, name=label_model1
    )
    
    # Plot Precision-Recall for Model 2
    disp2 = PrecisionRecallDisplay.from_predictions(
        y_test, y_pred_model2, name=label_model2, ax=disp1.ax_
    )
    
    # Add a title and legend
    plt.title("Precision-Recall Curve Comparison")
    plt.legend(loc="lower left")
    plt.show()
