import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve,
    auc, precision_recall_curve, average_precision_score, ConfusionMatrixDisplay
)
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_thresholds(y_true, probs, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """
    Evaluate model performance across different decision thresholds.
    """
    print("Threshold | Precision | Recall | F1-Score")
    print("------------------------------------------")
    for t in thresholds:
        preds = (probs >= t).astype(int)  # Convert probabilities to binary predictions
        precision = precision_score(y_true, preds)
        recall = recall_score(y_true, preds)
        f1 = f1_score(y_true, preds)
        print(f"   {t:.2f}    |   {precision:.3f}   |  {recall:.3f} |  {f1:.3f}")

def plot_precision_recall(y_true, probs):
    """
    Plot the Precision-Recall curve and save it to the 'plots' directory.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    avg_prec = average_precision_score(y_true, probs)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=f'AP = {avg_prec:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall Curve')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)  # Create 'plots' folder if it doesn't exist
    plt.savefig("plots/precision_recall_curve.png")
    plt.show()

def evaluate_model(model, X_test_tensor, y_test_tensor):
    """
    Evaluate a trained PyTorch model on the test set with various metrics and visualizations.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        logits = model(X_test_tensor)  # Get raw outputs
        probs = torch.sigmoid(logits).numpy().squeeze()  # Convert logits to probabilities
        preds = (probs >= 0.45).astype(int)  # Convert probabilities to binary predictions
        y_true = y_test_tensor.numpy().squeeze()  # Ground truth labels

    # Evaluate different thresholds to analyze model sensitivity
    print("\nEvaluation across different thresholds:")
    evaluate_thresholds(y_true, probs)

    # Show classification metrics (default threshold = 0.5)
    print("\nClassification Report (threshold=0.5):")
    print(classification_report(y_true, preds))

    # Plot confusion matrix
    cm = confusion_matrix(y_true, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Plot ROC curve and calculate AUC
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.legend()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/roc_curve.png")
    plt.show()

    # Plot Precision-Recall curve
    plot_precision_recall(y_true, probs)
