import torch
import os
from src.data_preparation import X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, X_train
from src.train import train_model
from src.evaluate import evaluate_model
from src.features_importance import plot_feature_importance
from src.tune_lr import tune_learning_rates
from src.cross_validation import cross_validate

# Configurations
input_dim = X_train_tensor.shape[1]
num_epochs = 2000

# Define a list of learning rates for testing
learning_rates = [0.001, 0.005, 0.01, 0.02]

# Searching best learning rate
print("----Searching for best learning rate:----\n")
best_lr = tune_learning_rates(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, input_dim, learning_rates, num_epochs=500)

# Training final model with the best learning rate
print(f"\n----Training final model with learning rate = {best_lr}:----\n")
model = train_model(X_train_tensor, y_train_tensor, input_dim, lr=best_lr, num_epochs=num_epochs, use_weight=True)

# Evaluating final model
print("\n----Evaluatingo final model:----\n")
evaluate_model(model, X_test_tensor, y_test_tensor)

# Plotting feature importance
plot_feature_importance(model, X_train)

# Cross validation
# This will give a better understanding of the model's performance across different splits of the data
print("\n----Cross Validation:----\n")
cross_validate(X_train_tensor, y_train_tensor, input_dim, lr=best_lr, num_epochs=1000, k=5)

# Saving the model
print("\n----Saving the model:----\n")
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/model.pth")
print("\nModel sucessfuly saved in 'models/model.pth'")
