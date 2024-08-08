import os
import pandas as pd

# Define the new base path pattern and the path for the output CSV
base_path_pattern = '../running-example/random_skill_each_category_{}/results/clustered_result_each_skill (kmeans)/'
output_csv_path_pattern = '../running-example/random_skill_each_category_{}/results/icon_inconsistency_result.csv'

# Initialize an empty list to store the data
results_list = []

# Process directories from 1
for i in range(1, 2): # change folder number here
    base_path = base_path_pattern.format(i)
    output_csv_path = output_csv_path_pattern.format(i)  # Format the output path with the current folder number

    # Check if the base path exists
    if os.path.exists(base_path):
        # List all CSV files that match the required pattern
        csv_files = [f for f in os.listdir(base_path) if f.startswith('cluster_result_') and f.endswith('.csv')]

        # Iterate through each CSV file
        for csv_file in csv_files:
            csv_path = os.path.join(base_path, csv_file)
            df = pd.read_csv(csv_path)  # Read the CSV file into a DataFrame

            # Extract ASIN from the file name
            asin = csv_file[len('cluster_result_'):-len('.csv')]

            # Read the first value in the 'Cluster' column to get the first cluster ID
            first_cluster_id = df['Cluster'].iloc[0]

            # Construct the expected Image_ID value
            expected_image_id = f"{asin}.png"

            # Check if the expected Image_ID exists in the DataFrame
            if expected_image_id in df['Image_ID'].values:
                # Extract the row where Image_ID matches the expected value
                matching_row = df[df['Image_ID'] == expected_image_id]
                matching_cluster_id = matching_row['Cluster'].iloc[0]

                # Determine consistency status as 0 for consistency, 1 for inconsistency
                consistency_status = 0 if first_cluster_id == matching_cluster_id else 1

                # Append the result to the results list
                results_list.append({'ASIN': asin, 'Consistency_Status': consistency_status})

    # Ensure the output directory exists before saving
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

# Convert list of dicts to DataFrame
results_df = pd.DataFrame(results_list)

# Save the results DataFrame to a new CSV file
results_df.to_csv(output_csv_path, index=False)
print("Icon inconsistency results have been saved.")
