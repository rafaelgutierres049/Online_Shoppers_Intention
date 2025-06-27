from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
from src.train import train_model 
import numpy as np

def cross_validate(X_tensor, y_tensor, input_dim, lr=0.001, num_epochs=1000, k=5):
    """
    k-fold Cross Validation in a logistic regression model with PyTorch

    Parameters:
    - X_tensor: Entry features tensor
    - y_tensor: Tensor with labels (0 or 1)
    - input_dim: Features quantities (number of columns in X_tensor)
    - lr: Learning rate for the optimizer
    - num_epochs: Number of epochs for training the model
    - k: Number of folds for cross-validation (default is 5)
    """
    
    # Creates a KFold object with k splits, shuffling the data before splitting
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Lists to store metrics for each fold
    # These will be used to calculate the average performance across all folds
    f1s, precisions, recalls, accs = [], [], [], []

    # Loop through each fold
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_tensor)):
        print(f"\n📦 Fold {fold+1}/{k}")
        
        # Separetes the data into training and testing sets for the current fold
        X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
        y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]

        # Trains the model using the training data of the current fold
        model = train_model(X_train, y_train, input_dim, lr=lr, num_epochs=num_epochs, use_weight=True)

        # Evaluates the model using the testing data of the current fold
        # Sets the model to evaluation mode, which disables dropout and batch normalization
        model.eval()
        with torch.no_grad():  
            logits = model(X_test)  
            probs = torch.sigmoid(logits).squeeze()  
            y_pred = (probs > 0.5).int().cpu().numpy() 
            y_true = y_test.cpu().numpy()  

        # Calculates the evaluation metrics for the current fold
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)

        # Appends the metrics to their respective lists
        f1s.append(f1)
        precisions.append(prec)
        recalls.append(rec)
        accs.append(acc)

        # Prints the metrics for the current fold
        print(f"F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, Accuracy: {acc:.4f}")

    # Calculates and prints the average metrics across all folds
    print("\n📊 Média dos resultados após validação cruzada:")
    print(f"F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
