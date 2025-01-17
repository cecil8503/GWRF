import pandas as pd  
import glob  
  
# 设置要合并的CSV文件的路径  
csv_folder_path = 'E:\\GWRF\\603\\1212\\FI'  # 替换为你的CSV文件所在的文件夹路径  
  
# 使用glob模块获取所有CSV文件的路径  
csv_files = glob.glob(csv_folder_path + '/*.csv')  
  
# 创建一个空的DataFrame，用于存储合并后的数据  
merged_data = pd.DataFrame()  
  
# 遍历每个CSV文件，并将其添加到merged_data中  
for file in csv_files:  
    df = pd.read_csv(file)  
    merged_data = pd.concat([merged_data, df], ignore_index=True)  
  
# 将合并后的数据保存到一个新的CSV文件中  
merged_data.to_csv('E:\\GWRF\\603\\1212FI.csv', index=False)  
  
print('所有CSV文件已成功合并到merged_file.csv中。')
