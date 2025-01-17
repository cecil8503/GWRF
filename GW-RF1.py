import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
np.random.seed(123)
pd.options.mode.chained_assignment = None
from multiprocessing import Pool, cpu_count


# MAPE计算
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# 定义加权方式，在这里更改带宽
def calculate_weights(data, point, bandwidth=3948.6925):
    weights = []
    # 遍历数据集中的每个点
    for _, row in data.iterrows():
        # 计算当前点与给定点的地理距离（米）
        distance = geodesic((row['latitude'], row['longitude']), point).meters
        #这里超过2倍带宽的被定义为0，如果带宽较大可考虑更改为一倍带宽
        if distance > 2 * bandwidth:
            weight = 0
        else:
        # 计算权重，这里使用高斯核加权，根据需要更改
            weight = np.exp(-np.square(distance) / (2 * np.square(bandwidth)))
        weights.append(weight)

    return np.array(weights)


# 单个任务的处理函数
def process_single_point(index, row, X_train, X_train_m, y_train, X_gwrf, X, y, feature_names):
    print(f"Processing index {index}")

    weights_train = calculate_weights(X_train, [row['latitude'], row['longitude']])
    weights = calculate_weights(X, [row['latitude'], row['longitude']])

    model = RandomForestRegressor(max_depth=20, random_state=123)
    model.fit(X_train_m, y_train, sample_weight=weights_train)

    cv = KFold(n_splits=5, shuffle=True, random_state=123)
    scoring = ['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error', 'explained_variance']
    cv_results = cross_validate(model, X_gwrf, y, cv=cv, scoring=scoring, fit_params={'sample_weight': weights})

    cv_rmse_scores = np.sqrt(-cv_results['test_neg_mean_squared_error'])
    Y_pred = model.predict(X_gwrf)[index]

    metrics = {
        'OBJECTID': row['OBJECTID'],
        'MAE': -np.mean(cv_results['test_neg_mean_absolute_error']),
        'RMSE': np.mean(cv_rmse_scores),
        'R2': np.mean(cv_results['test_r2']),
        'Explained Variance': np.mean(cv_results['test_explained_variance']),
        'Y_pred': Y_pred
    }

    feature_importance_dict = {'OBJECTID': row['OBJECTID']}
    for feature_name, importance in zip(feature_names, model.feature_importances_):
        feature_importance_dict[feature_name] = importance

    return metrics, feature_importance_dict


# 并行处理的主函数
def main_process(data, X_train, X_train_m, y_train, X_gwrf, X, y, feature_names, group=0):
    total_rows = len(data)
    model_metrics = []
    feature_importances_list = []

    with Pool(cpu_count()) as pool:
        tasks = []
        #注意这里只计算50行，防止崩溃后无法保存
        for i in range(5):
            for index, row in data.iterrows():
                if i * 10 + group <= index <= i * 10 + 9 + group:
                    tasks.append((index, row, X_train, X_train_m, y_train, X_gwrf, X, y, feature_names))

        # 使用pool.map并行处理
        results = pool.starmap(process_single_point, tasks)

        for metrics, feature_importance_dict in results:
            model_metrics.append(metrics)
            feature_importances_list.append(feature_importance_dict)

        # 保存结果
        model_metrics_df = pd.DataFrame(model_metrics)
        model_metrics_df.to_csv('E:\\GWRF\\model_metrics_{group}.csv', index=False)
        feature_importances_df = pd.DataFrame(feature_importances_list)
        feature_importances_df.to_csv('E:\\GWRF\\feature_importances_{group}.csv', index=False)


if __name__ == '__main__':
    # 导入xlsx
    file_path = 'E:\\GWRF\\BJ_8_Zscore.xlsx'
    # 读取 Excel 文件
    data = pd.read_excel(file_path)
    # 显示前几行数据以确认导入正确
    print(data.head())
    columns_to_drop = ['OBJECTID', 'LST']

    X = data.drop(columns=columns_to_drop)
    y = data["LST"]

    # 切分数据集后，你有了训练集和测试集的索引
    dis_to_drop = ['longitude', 'latitude']
    X_gwrf = X.drop(columns=dis_to_drop)
    feature_names = X_gwrf.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

    X_train_m = X_train.drop(columns=dis_to_drop)
    X_test_m = X_test.drop(columns=dis_to_drop)

    # 运行主进程，示例代码只运行前50行，通过更改group值选择运行的行数
    main_process(data, X_train, X_train_m, y_train, X_gwrf, X, y, feature_names, group=0)
