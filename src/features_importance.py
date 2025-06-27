import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_feature_importance(model, X_train):
    """
    Plots the importance of features based on the weights of a trained logistic regression model.

    Parameters:
    - model: Trained PyTorch logistic regression model (must have a .linear layer)
    - X_train: Training DataFrame (used for feature names)
    """

    # Extract the weights from the linear layer of the model and flatten the array
    weights = model.linear.weight.detach().numpy().flatten()

    # Get the feature names from the training DataFrame
    feature_names = X_train.columns

    # Create a DataFrame with feature names and their corresponding weights
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': weights
    })

    # Add a column with absolute importance (for sorting)
    importance_df['AbsImportance'] = importance_df['Importance'].abs()

    # Sort features by absolute importance (low to high)
    importance_df = importance_df.sort_values(by='AbsImportance', ascending=True)

    # Create horizontal bar plot
    plt.figure(figsize=(10, 12))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.axvline(0, color='gray', linestyle='--')  # Reference line at zero
    plt.xlabel("Logistic Regression Weight")
    plt.title("Feature Importance (Model Coefficients)")
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save the plot to the 'plots' folder
    os.makedirs("plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png")
    plt.show()
