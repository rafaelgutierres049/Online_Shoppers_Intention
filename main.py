import torch
from src.data_preparation import get_tensors
from src.train import train_model
from src.evaluate import evaluate_model

X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = get_tensors()

input_dim = X_train_tensor.shape[1]
model = train_model(X_train_tensor, y_train_tensor, input_dim)

evaluate_model(model, X_test_tensor, y_test_tensor)

# Save the model
torch.save(model.state_dict(), 'saved_model/logistic_regression_model.pth')