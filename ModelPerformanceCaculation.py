from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import pandas as pd 
# 用于计算的csv应该至少有两列数据：'y-pred'预测值, 'y'实际值

data = pd.read_csv('储存结果的文件.csv') 

# 提取预测值和实际值
y_pred = data['y-pred']
y_true = data['y']

# 计算R2, MAE, RMSE, Explained Variance
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
explained_variance = explained_variance_score(y_true, y_pred)

# 打印结果
r2, mae, rmse, explained_variance
