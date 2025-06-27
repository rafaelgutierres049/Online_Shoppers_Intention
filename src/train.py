import torch
import torch.nn as nn
import torch.optim as optim
from src.logistic_model import LogisticRegressionModel

def train_model(X_train_tensor, y_train_tensor, input_dim, lr, num_epochs=2000, use_weight=False):
    """
    Train a logistic regression model using BCEWithLogitsLoss and Adam optimizer.

    Parameters:
    - X_train_tensor: Input features tensor for training (shape: [num_samples, input_dim])
    - y_train_tensor: Target labels tensor for training (shape: [num_samples, 1] or [num_samples])
    - input_dim: Number of input features (dimensionality)
    - lr: Learning rate for the optimizer
    - num_epochs: Number of epochs to train the model
    - use_weight: Boolean flag to apply positive class weighting in the loss function (to handle class imbalance)

    Returns:
    - Trained model (instance of LogisticRegressionModel)
    """
    # Initialize the logistic regression model
    model = LogisticRegressionModel(input_dim)
    
    # Compute positive class weight if requested, to address class imbalance
    if use_weight:
        # Count positive and negative samples
        num_pos = y_train_tensor.sum()
        num_neg = y_train_tensor.shape[0] - num_pos
        
        # Calculate pos_weight as ratio of negatives to positives
        pos_weight = num_neg / num_pos
        
        # Convert pos_weight to tensor if not already one (detach to avoid tracking gradients)
        pos_weight_tensor = pos_weight.clone().detach() if isinstance(pos_weight, torch.Tensor) else torch.tensor(pos_weight, dtype=torch.float32)
        
        # Use BCEWithLogitsLoss with pos_weight to emphasize positive samples
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        # Standard BCEWithLogitsLoss without weighting
        criterion = nn.BCEWithLogitsLoss()
    
    # Adam optimizer with L2 regularization via weight_decay parameter
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    for epoch in range(num_epochs):
        model.train()            # Set model to training mode
        optimizer.zero_grad()    # Clear gradients
        
        logits = model(X_train_tensor)  # Forward pass: raw logits output (no sigmoid)
        
        loss = criterion(logits, y_train_tensor)  # Compute loss
        
        loss.backward()          # Backpropagation
        optimizer.step()         # Update model parameters
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}")
    
    return model
