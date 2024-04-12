# # # Define a function to extract 'lambdaLog' lines and write them to a new file
# # def extract_lambda_logs(file_path, output_file_path):
# #     with open(file_path, 'r') as file, open(output_file_path, 'w') as output_file:
# #         for line in file:
# #             if 'lambdaLog' in line:
# #                 output_file.write(line)

# # # Call the function to extract 'lambdaLog' lines from 'logfile.log' and write them to 'lambda_logs.txt'
# # extract_lambda_logs('logfile.log', 'lambda_logs.txt')
# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the CSV file into a DataFrame
# df = pd.read_csv('e91efa9b-0162-410b-bf18-d2e50e9d0f2a.csv')

# # Get unique products
# products = df['product'].unique()

# # Create a figure and axes for subplots
# fig, ax = plt.subplots(len(products), 2, figsize=(10, 8))

# # Plot bid_price_1 and ask_price_1 for each product
# for i, product in enumerate(products):
#     # Filter data for the current product
#     product_data = df[df['product'] == product]

#     # Plot bid_price_1
#     ax[i][0].plot(product_data['bid_price_1'], label='Bid Price 1', color='blue')
#     ax[i][0].set_title(f'Bid Price 1 for {product}')
#     ax[i][0].set_xlabel('Index')
#     ax[i][0].set_ylabel('Price')
#     ax[i][0].legend()

#     # Plot ask_price_1
#     ax[i][1].plot(product_data['ask_price_1'], label='Ask Price 1', color='red')
#     ax[i][1].set_title(f'Ask Price 1 for {product}')
#     ax[i][1].set_xlabel('Index')
#     ax[i][1].set_ylabel('Price')
#     ax[i][1].legend()

# # Adjust layout and show plot
# plt.tight_layout()
# plt.show()

import json

# Function to convert a JSON file into a Python dictionary
def json_to_dict(json_file_path):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

# Specify the path to your JSON file
json_file_path = 'logfile.json'

# Call the function and print the result
data = json_to_dict(json_file_path)
print(data)


