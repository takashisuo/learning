import matplotlib
matplotlib.use('AGG')
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns


def clusterpy(file_path):
    df = pd.read_excel(file_path)
    df = df.rename(
        columns = {
            '輸送重量': 'WGT',
            '輸送容量': 'VOL',
            '輸送車数': 'Truck',
            '輸送コスト': 'Cost'
            },
        )
    df["日付"] = df['日付'].dt.date

    k_means = KMeans(n_clusters=4)
    cluster = k_means.fit_predict(df.values[:, 2:])

    df['cluster'] = ['cluster_'+str(i) for i in cluster]

    df = df.sort_values('cluster')
    df = df.reindex(
        columns=['日付', '出発・到着', 'cluster', 'WGT', 'VOL', 'Truck', 'Cost']
        )
    
    result_path = 'download/' + file_path[7:-5]
    print(f"path: {result_path}")

    sns.pairplot(
        df, hue='cluster', vars=['WGT', 'VOL', 'Truck', 'Cost']
        ).savefig(result_path + '_cluter.png')
    
    df.to_excel(result_path + '_cluster.xlsx', index=False)


if __name__ == '__main__':
    clusterpy('sample.xlsx')