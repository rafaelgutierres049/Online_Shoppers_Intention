import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

data = pd.read_csv('data/online_shoppers_intention.csv')

# print(data.head())
# print(data.info())
# print(data['Revenue'].value_counts())

data['Revenue'] = data['Revenue'].astype(int)
data['Weekend'] = data['Weekend'].astype(int)

#One hot encoding for categorical variables
data = pd.get_dummies(data, columns=['Month', 'VisitorType'], drop_first=True)

X = data.drop(['Revenue'], axis=1)
y = data['Revenue']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
y_test_scaled = scaler.transform(X_test)

#tensor conversion
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)    
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

def get_tensors():
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

# print(f'X_train_tensor shape: {X_train_tensor.shape}')
# print(f'y_train_tensor shape: {y_train_tensor.shape}')