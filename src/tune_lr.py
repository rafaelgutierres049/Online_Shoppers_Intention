import torch
import torch.nn as nn
import torch.optim as optim
from src.logistic_model import LogisticRegressionModel
from sklearn.metrics import f1_score

def train_model_simple(X_train_tensor, y_train_tensor, input_dim, lr=0.001, num_epochs=500):
    """
    Train a logistic regression model using BCEWithLogitsLoss and Adam optimizer.

    Args:
        X_train_tensor: Training features tensor.
        y_train_tensor: Training labels tensor.
        input_dim: Number of input features.
        lr: Learning rate.
        num_epochs: Number of training epochs.

    Returns:
        Trained LogisticRegressionModel instance.
    """
    model = LogisticRegressionModel(input_dim)
    criterion = nn.BCEWithLogitsLoss()  # Combines sigmoid + BCE loss for numerical stability
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)  # raw logits output
        loss = criterion(outputs.squeeze(), y_train_tensor.squeeze())  # squeeze in case tensors have extra dims
        loss.backward()
        optimizer.step()

    return model

def tune_learning_rates(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, input_dim, lrs, num_epochs=500):
    """
    Perform hyperparameter tuning over a list of learning rates.

    Args:
        X_train_tensor, y_train_tensor: Training data.
        X_test_tensor, y_test_tensor: Validation/test data.
        input_dim: Number of input features.
        lrs: List of learning rates to test.
        num_epochs: Training epochs for each lr.

    Returns:
        The best learning rate based on validation F1 score.
    """
    best_lr = None
    best_f1 = 0.0

    for lr in lrs:
        print(f"\nTraining with learning rate = {lr}")
        model = train_model_simple(X_train_tensor, y_train_tensor, input_dim, lr=lr, num_epochs=num_epochs)

        model.eval()
        with torch.no_grad():
            logits = model(X_test_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()  # ensure on CPU and squeeze dims
            preds = (probs >= 0.5).astype(int)
            y_true = y_test_tensor.cpu().numpy().squeeze()

        f1 = f1_score(y_true, preds)
        print(f"F1-Score at lr={lr}: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_lr = lr

    print(f"\nBest learning rate: {best_lr} with F1-Score: {best_f1:.4f}")
    return best_lr
