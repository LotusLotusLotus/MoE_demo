import pandas as pd

from data import generate_csv

# Generate the CSV (using the previously provided generate_csv function)
generate_csv('data.csv')

# Read and display the first 10 rows of the CSV
dataframe = pd.read_csv('data.csv')
print(dataframe.head(10))