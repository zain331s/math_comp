import torch  # PyTorch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split  # Splits data into training/testing sets
from sklearn.preprocessing import StandardScaler  # Scales data for better performance

df = pd.read_csv("EditedCSVs/df2CSV.csv")
print(df.head()) 

df["Medal"] = df["Medal"].astype('category').cat.codes

df1 = df.drop(["Medal"], axis=1)  
X = df1.to_numpy()
y = df["Medal"].to_numpy()

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Neural Network Definition
class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 15)
        self.fc4 = nn.Linear(15, output_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# Train-Test Split and Scaling
def train_split(X, y):
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

# Training Function
def train(X_train_tensor, y_train_tensor, model, num_epochs=50, learning_rate=0.01, filename=None):
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train_tensor)
        loss = loss_fn(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Save the model if a filename is provided
    if filename:
        torch.save(model.state_dict(), filename)
        print(f"Model saved as: {filename}")

# Accuracy Calculation
def compute_accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predictions = torch.max(outputs, 1)
        accuracy = (predictions == y).float().mean().item()
    return accuracy

# Main Script
input_size = X.shape[1]
output_size = len(np.unique(y))
X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = train_split(X, y)

model = SimpleNeuralNetwork(input_size, output_size)
train(X_train_tensor, y_train_tensor, model, num_epochs=200, filename="model.pth")

# Evaluate Model
train_acc = compute_accuracy(model, X_train_tensor, y_train_tensor)
test_acc = compute_accuracy(model, X_test_tensor, y_test_tensor)
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")


# Example input data
example_data = [[1, 3, 5, 5, 5, 6]]  # Replace with the feature values for your example

# Scale the input data using the same scaler used during training
scaler = StandardScaler()
scaler.fit(X)  # Fit the scaler to the full dataset
example_data_scaled = scaler.transform(example_data)

# Convert the scaled data to a PyTorch tensor
example_data_tensor = torch.tensor(example_data_scaled, dtype=torch.float32).to(device)

# Make predictions
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    output = model(example_data_tensor)
    _, predicted_class = torch.max(output, 1)  # Get the predicted class index

# Interpret the result
class_names = ["Bronze", "Silver", "Gold"]  # Replace with your actual class names if different
print(f"Predicted class index: {predicted_class.item()}")
print(f"Predicted class: {class_names[predicted_class.item()]}")
