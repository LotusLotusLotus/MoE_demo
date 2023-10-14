import csv
import random

def generate_data_row():
    # Randomly decide if this sample should be all odd or all even
    if random.random() > 0.5:  # Generate odd numbers
        data = [random.randint(1, 5) * 2 - 1 for _ in range(5)]  # Odd numbers between 1 and 10
        target = sum(data)
        gate_label = 0
    else:  # Generate even numbers
        data = [random.randint(1, 5) * 2 for _ in range(5)]  # Even numbers between 2 and 10
        target = sum(data[:4]) - data[4]
        gate_label = 1

    return data, target, gate_label

def generate_csv(filename, num_samples=1000):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow(['input1', 'input2', 'input3', 'input4', 'input5', 'targetdata', 'gate_label'])
        # Write data rows
        for _ in range(num_samples):
            data, target, gate_label = generate_data_row()
            csvwriter.writerow(data + [target, gate_label])

# Generate the CSV
generate_csv('data.csv')