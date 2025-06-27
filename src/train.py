import torch.nn as nn
import torch.optim as optim
from src.logistic_model import LogisticRegressionModel

def train_model(X_train_tensor, y_train_tensor, input_dim, lr=0.01, num_epochs=1000):
    model = LogisticRegressionModel(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}")
    
    return model
