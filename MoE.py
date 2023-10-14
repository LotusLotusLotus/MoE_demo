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

# Gating Network as Classifier
class GatingNetwork(nn.Module):
    def __init__(self, input_dim):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

# Data Loading from CSV
def load_data_from_csv(filename):
    dataframe = pd.read_csv(filename)
    inputs = torch.tensor(dataframe[['input1', 'input2', 'input3', 'input4', 'input5']].values).float()
    labels = torch.tensor(dataframe[['target_data']].values).float()
    gate_labels = torch.tensor(dataframe[['gate_label']].values).float()
    return inputs, labels, gate_labels

# Training Parameters
num_epochs = 10000
learning_rate = 0.001

expert_odd = ExpertModel(5)
expert_even = ExpertModel(5)
gating = GatingNetwork(5)

criterion = nn.MSELoss()
criterion_gate = nn.BCELoss()

optimizer = optim.Adam(list(expert_odd.parameters()) + list(expert_even.parameters()), lr=learning_rate)
optimizer_gate = optim.Adam(gating.parameters(), lr=learning_rate)

# Training Gating Network
for epoch in range(num_epochs // 2):
    inputs, _, gate_labels = load_data_from_csv('data.csv')
    optimizer_gate.zero_grad()
    gate_outputs = gating(inputs)
    loss_gate = criterion_gate(gate_outputs, gate_labels)
    loss_gate.backward()
    optimizer_gate.step()


# Training Expert Models
for epoch in range(num_epochs // 2, num_epochs):
    inputs, labels, gate_labels = load_data_from_csv('data.csv')

    # Odd data for expert_odd
    odd_mask = (gate_labels == 0).float()
    output_odd = expert_odd(inputs)
    loss_odd = criterion(output_odd * odd_mask, labels * odd_mask)

    optimizer.zero_grad()
    loss_odd.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Expert Odd Loss: {loss_odd.item():.4f}')

    # Even data for expert_even
    even_mask = (gate_labels == 1).float()
    output_even = expert_even(inputs)
    loss_even = criterion(output_even * even_mask, labels * even_mask)

    optimizer.zero_grad()
    loss_even.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Expert Even Loss: {loss_even.item():.4f}')

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

# Compute the predictions and test loss
with torch.no_grad():
    gate_outputs = (gating(test_data_tensor) > 0.5).float()
    output_odd = expert_odd(test_data_tensor)
    output_even = expert_even(test_data_tensor)
    outputs = (1 - gate_outputs) * output_odd + gate_outputs * output_even
    test_loss = criterion(outputs, torch.Tensor(target_data).view(-1, 1))

print(f"Test Loss: {test_loss.item():.4f}")

# Output the predictions
for i, sample in enumerate(test_data):
    expert_used = "Expert Odd" if gate_outputs[i].item() == 0 else "Expert Even"
    print(f"Prediction for {sample.tolist()}: {outputs[i].item():.4f} (Used {expert_used})")


