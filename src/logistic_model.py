import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        """
        Initialize the logistic regression model.

        Parameters:
        - input_dim: Number of input features (dimensionality of the input data)
        """
        super(LogisticRegressionModel, self).__init__()

        # Define a single linear layer that maps input features to a single output (logit)
        # This layer performs a weighted sum of input features plus a bias term
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
        - x: Input tensor with shape (batch_size, input_dim)

        Returns:
        - Logits (raw scores) without applying any activation function such as sigmoid.
          This is typical for binary classification tasks where
          a loss function like BCEWithLogitsLoss applies sigmoid internally.
        """
        return self.linear(x)  # Output raw logits; sigmoid will be applied externally if needed
