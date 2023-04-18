import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering, DBSCAN, KMeans, AgglomerativeClustering
from skfuzzy.cluster import cmeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score  # 轮廓系数
import scikitplot as skplt
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib
from sklearn.decomposition import PCA
import copy
import os

plt.style.use(['science', 'grid', 'no-latex'])
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1

# 设置保存图片的格式和dpi
matplotlib.rcParams['savefig.dpi'] = 600
matplotlib.rcParams['savefig.format'] = 'svg'

# 调用scikitplot绘图


def plot_silhouette_of_various_clusters(num_li, PCA_behavior_features, data_type, sheet_name):
    for i in num_li:
        skplt.metrics.plot_silhouette(PCA_behavior_features,
                                      SpectralClustering(n_clusters=i, random_state=0, affinity='rbf').fit_predict(
                                          PCA_behavior_features),
                                      figsize=(6, 6), title=None)
        plt.xlim(-0.1, 0.5)
        plt.savefig(
            f'./asset/{data_type}/sheet{sheet_name}/figure/谱聚类聚类为{i}簇时各样本的轮廓系数值', bbox_inches='tight')


def get_ratio(input_n_clusters, PCA_behavior_features, data_type, sheet_name):
    clusterer = SpectralClustering(
        n_clusters=input_n_clusters, random_state=0, affinity='rbf').fit_predict(PCA_behavior_features)
    # print(f"聚类为{input_n_clusters}类")
    # print(pd.Series(clusterer).value_counts() / clusterer.shape[0])
    df_add_label = pd.concat([pd.DataFrame(PCA_behavior_features), pd.DataFrame(
        [*clusterer], columns=['Cluster'])], axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))  # 设定画布
    for label, temp_df in df_add_label.groupby('Cluster'):
        # print(temp_df)
        ax.scatter(temp_df.loc[:, 0], temp_df.loc[:, 1],
                   color=COLOR_LIST[int(label)], s=70)
        # ax.plot(DBS_Score['n_clusters'],DBS_Score[it],lw=1.2,marker='o',ms=8,label=it,ls='--',color=COLOR_LIST[idx]) #绘制每一条曲线
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    # ax.legend(fontsize=13, frameon=True, loc='best')
    plt.savefig(
        f'./asset/{data_type}/sheet{sheet_name}/figure/聚类为{input_n_clusters}簇下的散点分布图', bbox_inches='tight')
    return dict(pd.Series(clusterer).value_counts() / clusterer.shape[0])


def main(data_path, sheet_name, data_type="风电"):
    # 判断路径是否存在，不存在就生成
    figure_path = f"./asset/{data_type}/sheet{sheet_name}/figure"
    result_path = f"./asset/{data_type}/sheet{sheet_name}/data"
    if not os.path.exists(figure_path):  #如果路径不存在
        os.makedirs(figure_path, exist_ok=True)
        print(f"路径【{result_path}】生成成功")
    if not os.path.exists(result_path):  #如果路径不存在
        os.makedirs(result_path, exist_ok=True)

    # 读取数据
    df = pd.read_excel(data_path, sheet_name=0, index_col=0).T
    copyResult = copy.deepcopy(df)

    # 数据归一化
    
    scaler = MinMaxScaler()  # 实例化
    scaler = scaler.fit(copyResult)  # fit，在这里本质是生成min(x)和max(x)
    MinMaxScaler_features = scaler.transform(copyResult)  # 通过接口导出结果

    # PCA降维
    pca = PCA(n_components=0.85, svd_solver="full")  # 实例化
    pca = pca.fit(MinMaxScaler_features)  # 拟合模型
    PCA_behavior_features = pca.transform(MinMaxScaler_features)

    # 绘图
    skplt.decomposition.plot_pca_component_variance(
        pca, target_explained_variance=0.85, figsize=(6, 6), title=None)
    plt.savefig(f'./asset/{data_type}/sheet{sheet_name}/figure/PCA累积贡献率', bbox_inches='tight')

    # 计算聚类的评价指标
    CH_Score = []
    Sil_Score = []
    DBS_Score = []
    for coef in range(2, 9):
        # 追加CH分数Calinski Harabasz Score
        Sil_Score.append([coef, silhouette_score(PCA_behavior_features,
                                                 KMeans(n_clusters=coef, random_state=10).fit(PCA_behavior_features).labels_),
                          silhouette_score(PCA_behavior_features,
                                           SpectralClustering(n_clusters=coef, random_state=0, affinity='rbf').fit_predict(PCA_behavior_features))
                          ])

        # 追加轮廓系数
        CH_Score.append([coef, calinski_harabasz_score(PCA_behavior_features,
                                                       KMeans(n_clusters=coef, random_state=10).fit(PCA_behavior_features).labels_),
                         calinski_harabasz_score(PCA_behavior_features,
                                                 SpectralClustering(n_clusters=coef, random_state=0, affinity='rbf').fit_predict(PCA_behavior_features))
                         ])

        # 追加DBS系数
        DBS_Score.append([coef, davies_bouldin_score(PCA_behavior_features,
                                                     KMeans(n_clusters=coef, random_state=10).fit(PCA_behavior_features).labels_),
                          davies_bouldin_score(PCA_behavior_features, SpectralClustering(
                              n_clusters=coef, random_state=0, affinity='rbf').fit_predict(PCA_behavior_features))
                          ])

    CH_Score = pd.DataFrame(
        CH_Score, columns=['n_clusters', 'K-Means++', 'SpectralClustering'])
    Sil_Score = pd.DataFrame(
        Sil_Score, columns=['n_clusters', 'K-Means++', 'SpectralClustering'])
    DBS_Score = pd.DataFrame(
        DBS_Score, columns=['n_clusters', 'K-Means++', 'SpectralClustering'])
    # 导出数据到本地
    CH_Score.to_csv(f'./asset/{data_type}/sheet{sheet_name}/data/CH_Score.csv',
                    index=False, encoding='utf_8_sig')
    Sil_Score.to_csv(f'./asset/{data_type}/sheet{sheet_name}/data/Sil_Score.csv',
                     index=False, encoding='utf_8_sig')
    DBS_Score.to_csv(f'./asset/{data_type}/sheet{sheet_name}/data/DBS_Score.csv',
                     index=False, encoding='utf_8_sig')

    # 绘制聚类效果对比曲线(1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))  # 设定画布
    for idx, it in enumerate(['K-Means++', 'SpectralClustering']):
        ax1.plot(CH_Score['n_clusters'], CH_Score[it], lw=1.2, marker='o',
                 ms=8, label=it, ls='--', color=COLOR_LIST[idx])  # 绘制每一条曲线
        ax2.plot(Sil_Score['n_clusters'], Sil_Score[it], lw=1.2,
                 marker='^', ms=8, label=it, ls='--', color=COLOR_LIST[idx])
    ax1.set_xlabel('n_clusters')
    ax1.set_ylabel('calinski_harabasz_score')
    ax1.legend(fontsize=13, frameon=True, loc='best')
    ax2.set_xlabel('n_clusters')
    ax2.set_ylabel('silhouette_score')
    # ‘upper left’, ‘upper right’, ‘lower left’, ‘lower right’
    ax2.legend(fontsize=13, frameon=True, loc='best')
    plt.savefig(f'./asset/{data_type}/sheet{sheet_name}/figure/不同聚类方法的轮廓系数值对比', bbox_inches='tight')

    # 绘制聚类效果对比曲线(2)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))  # 设定画布
    for idx, it in enumerate(['K-Means++', 'SpectralClustering']):
        ax.plot(DBS_Score['n_clusters'], DBS_Score[it], lw=1.2, marker='o',
                ms=8, label=it, ls='--', color=COLOR_LIST[idx])  # 绘制每一条曲线
    ax.set_xlabel('n_clusters')
    ax.set_ylabel('davies_bouldin_score')
    ax.legend(fontsize=13, frameon=True, loc='best')
    plt.savefig(f'./asset/{data_type}/sheet{sheet_name}/figure/不同聚类方法的轮廓系数值对比-', bbox_inches='tight')

    # 调用scikitplot绘图
    plot_silhouette_of_various_clusters([*range(2, 9)], PCA_behavior_features, data_type, sheet_name)

    # 计算各类样本占比
    ratio_list = []
    for idx in range(2,9):
        ratio_dict = get_ratio(idx, PCA_behavior_features, data_type, sheet_name)
        ratio_list.append(ratio_dict)
    ratio_df = pd.DataFrame(ratio_list)
    ratio_df.to_csv(f'./asset/{data_type}/sheet{sheet_name}/data/ratio_df.csv',
                    index=False, encoding='utf_8_sig')


if __name__ == "__main__":
    COLOR_LIST = ['tab:red', 'black', "tab:blue", "tab:orange", "tab:green", "tab:purple",
                  "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan", "royalblue"]
    for sheet_index in range(12):
        main(data_path="./datasets/风电.xlsx", sheet_name=sheet_index, data_type="风电")
        main(data_path="./datasets/光伏.xlsx", sheet_name=sheet_index, data_type="光伏")
