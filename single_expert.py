import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Expert Model
class ExpertModel(nn.Module):
    def __init__(self, input_dim):
        super(ExpertModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)

# Data Loading from CSV
def load_data_from_csv(filename):
    dataframe = pd.read_csv(filename)
    inputs = torch.tensor(dataframe[['input1', 'input2', 'input3', 'input4', 'input5']].values).float()
    labels = torch.tensor(dataframe[['target_data']].values).float()
    return inputs, labels

# Training Parameters
num_epochs = 10000
learning_rate = 0.001

expert = ExpertModel(5)
criterion = nn.MSELoss()
optimizer = optim.Adam(expert.parameters(), lr=learning_rate)

# Training Expert Model
for epoch in range(num_epochs):
    inputs, labels = load_data_from_csv('data.csv')

    outputs = expert(inputs)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Define the test data and corresponding target data
test_data = [
    torch.Tensor([[1, 3, 5, 7, 9]]),
    torch.Tensor([[2, 6, 8, 10, 4]]),
    torch.Tensor([[3, 5, 7, 9, 1]]),
    torch.Tensor([[4, 6, 8, 10, 2]]),
    torch.Tensor([[5, 7, 9, 1, 3]]),
    torch.Tensor([[1, 1, 1, 1, 1]]),
    torch.Tensor([[2, 2, 2, 10, 4]]),
    torch.Tensor([[3, 7, 7, 7, 1]]),
    torch.Tensor([[4, 6, 8, 10, 2]]),
    torch.Tensor([[5, 7, 5, 1, 5]]),
]

target_data = [
    25,  # 1 + 3 + 5 + 7 + 9
    22,  # 2 + 6 + 8 + 10 - 4
    25,  # 3 + 5 + 7 + 9 + 1
    26,  # 4 + 6 + 8 + 10 - 2
    25,  # 5 + 7 + 9 + 1 + 3
    5,   # 1 + 1 + 1 + 1 + 1
    12,  # 2 + 2 + 2 + 10 - 4
    25,  # 3 + 7 + 7 + 7 + 1
    26,  # 4 + 6 + 8 + 10 - 2
    23   # 5 + 7 + 5 + 1 + 5
]

# Convert the list of tensors into a single tensor
test_data_tensor = torch.cat(test_data, dim=0)

# Compute the predictions and test loss using the single expert
with torch.no_grad():
    outputs = expert(test_data_tensor)  # Using expert_odd as the single expert
    test_loss = criterion(outputs, torch.Tensor(target_data).view(-1, 1))

print(f"Test Loss: {test_loss.item():.4f}")

# Output the predictions
for i, sample in enumerate(test_data):
    print(f"Prediction for {sample.tolist()}: {outputs[i].item():.4f}")



