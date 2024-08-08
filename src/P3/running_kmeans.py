import pandas as pd
from sklearn.cluster import KMeans
import os

# k=3
for folder_number in range(1,2):
  K = 3
  p = f'../running-example/random_skill_each_category_{folder_number}/results/distance_matrix_all_together.csv'
  df_sample_reload = pd.read_csv(p)  

  output_dir = f'../running-example/random_skill_each_category_{folder_number}/results/clustered_result_each_skill (kmeans)/'
  os.makedirs(output_dir, exist_ok=True)
  
  for index, row in df_sample_reload.iterrows():
      text_id = row.iloc[0]
      row_data = row.iloc[1:].values.reshape(-1, 1)
      
      kmeans = KMeans(n_clusters=K, random_state=0).fit(row_data)
      labels = kmeans.labels_
      
      result_df_reload = pd.DataFrame({
          'Image_ID': row.index[1:],
          'Cluster': labels,
          'Distance': row.iloc[1:].values  # 保存距离值
      })


      result_df_sorted = result_df_reload.sort_values(by='Distance', ascending=False)
      
      result_csv_path = os.path.join(output_dir, f'cluster_result_{text_id}.csv')
      result_df_sorted.to_csv(result_csv_path, index=False)

  # 提示输出目录路径
  print(f"Clustered rows CSV files are saved in: {output_dir}")
