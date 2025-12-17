import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
from datetime import datetime
import logging
import io
import time

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"outputs/autoencoding_kmeans_{timestamp}"
# 檢查資料夾是否存在，不存在就建立
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"已建立資料夾：{output_dir}")
else:
    print(f"資料夾已存在：{output_dir}")

# 建立 timestamp log 檔案
log_path = os.path.join(output_dir, f"run_{timestamp}.log")

# 設定 logging
logging.basicConfig(
    level=logging.INFO,  # 可改 DEBUG / WARNING / ERROR
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler()  # 同時輸出到 console
    ]
)
logger = logging.getLogger(__name__)
# ====== 記錄整體開始時間 ======
experiment_start = time.time()
logger.info("===== Experiment started =====")
logger.info(f"Logging started. Output file: {log_path}")

class Timer:
    def __init__(self, logger):
        self.logger = logger
        self.start = time.time()

    def log(self, stage_name):
        now = time.time()
        elapsed = now - self.start
        self.start = now  # 重置起點

        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60

        self.logger.info(
            f"{stage_name} finished in {hours}h {minutes}m {seconds:.2f}s"
        )

# 建立 Timer 
timer = Timer(logger)
# 降低 TensorFlow 的 log 等級（避免太多資訊干擾）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 檢查是否有可用的 GPU
logger.info("Num GPUs Available: %d", len(tf.config.list_physical_devices('GPU')))

# 讀取 CSV 文件，將數據加載到 DataFrame 中
df = pd.read_csv("./datasets/dataset.csv")
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
logger.info("DataFrame info:\n%s", info_str)

# -----------------------------
# 檢查缺失值
# -----------------------------
missing_cols = df.isna().sum()
missing_cols = missing_cols[missing_cols > 0]

for col, missing in missing_cols.items():
    logger.info(f"{col}: {missing} missing")

'''
定義音訊特徵與 metadata 特徵
audio_features：Spotify 的 9 個連續型音訊特徵
meta_features：歌曲的歌手、流派、熱門度
移除重複歌曲
避免同名歌曲造成模型偏差。
將類別特徵轉換為數值（Label Encoding）
Autoencoder 與 KMeans 無法處理字串，因此：

artists → artists_le
track_genre → genre_le
建立最終特徵集合 all_features
包含：

9 個 audio features
popularity（數值）
artists_le（編碼後的歌手）
genre_le（編碼後的曲風
'''
audio_features = [
    'danceability', 'energy', 'valence', 'liveness', 'acousticness',
    'instrumentalness', 'speechiness', 'tempo', 'loudness'
    ]
meta_features = ['artists','popularity','track_genre']

# 建立副本，避免修改原始 df
df = df.copy()

# -----------------------------
# 檢查缺失值
# -----------------------------
missing_cols = df.isna().sum()
missing_cols = missing_cols[missing_cols > 0]

for col, missing in missing_cols.items():
    print(f"{col}: {missing} missing")

# 補缺失值（避免 NaN 造成 drop_duplicates 行為不一致）
logger.info("Filling missing values for 'track_name' and 'artists'")
df['track_name'] = df['track_name'].fillna("Unknown Track")
df['artists'] = df['artists'].fillna("Unknown Artist")

# 依 track_name & artists 刪除重複歌曲，保留第一次出現的記錄，避免推薦系統推薦重複歌曲
logger.info("Dropping duplicate songs based on 'track_name' and 'artists'")
df = df.drop_duplicates(subset=['track_name','artists'], keep='first').reset_index(drop=True)

'''
同時印出總筆數與 unique 數量 對後續做 Label Encoding、Embedding、Autoencoder 都很有幫助，因為可以快速知道：
- artists 是否為 high-cardinality 特徵
- 是否需要避免 One-Hot
- 是否需要 embedding layer 或 autoencoder
'''
unique_artists = df['artists'].unique().tolist()
logger.info("Artists: count=%d", len(unique_artists))
unique_genres = df['track_genre'].unique()
logger.info("Genres: count=%d, name=%s", len(unique_genres), unique_genres)

'''
Label Encoding 的使用時機
1. 類別本身有順序（ordinal）
2. 類別數量極大（high cardinality）
'''
logger.info("Label encoding 'artists'")
# Label encode artists
le_artist = LabelEncoder()
df['artists_le'] = le_artist.fit_transform(df['artists'])

# # Label encode track_genre
# le_genre = LabelEncoder()
# df['genre_le'] = le_genre.fit_transform(df['track_genre'])

'''
One-Hot Encoding 的使用時機
1. 類別沒有順序（nominal）One-Hot 能讓 Autoencoder 自己學到類別之間的語意距離，而不會被假順序干擾。
2. 類別數量不大（< 1000）One-Hot 維度不會太大，Autoencoder 可以有效壓縮。
3. 你希望 Autoencoder 學到「類別之間的相似性」
'''

'''
同時對 artists 與 genre 做 one-hot encoding 會導致維度過高，記憶體爆炸
'''
logger.info("One-Hot encoding 'track_genre'")
# ===== One-Hot Encoding for genre =====
ohe = OneHotEncoder(sparse_output=False)

# 做 One-Hot
ohe_features = ohe.fit_transform(df[['track_genre']])

# ===== PCA 壓縮成 embedding（你可調整維度）=====
pca = PCA(n_components=5, random_state=42)
emb = pca.fit_transform(ohe_features)

# ===== 加回 DataFrame =====
emb_cols = [f"genre_emb_{i}" for i in range(5)]
df[emb_cols] = emb

# ===== 產生 genre_le（保留給後續程式碼使用）=====
# 用 embedding 的第一維當作 genre_le（可排序、有語意）
df['genre_le'] = df['genre_emb_0']

# 最終特徵集合
all_features = audio_features + ['popularity', 'artists_le', 'genre_le']


'''
Scaling + PCA（先看資料結構）
這裡的 PCA 先當作「資料探索用」，用來確認音樂特徵是否有可分群的結構
(1) 資料是否呈現出自然的群集？ 例如：

有明顯的 blob 形狀大致圓或橢圓的資料點群 → KMeans 可能很好用
呈現長條狀 / 月牙形 → KMeans 不適合，HDBSCAN / UMAP 會更好
完全一團 → 可能需要非線性 embedding（Autoencoder / UMAP）
(2) 是否有離群點？

若有明顯孤立點 → clustering 會受影響
可能需要先做 outlier removal
(3) 是否有線性可分性？

若資料呈現線性方向 → PCA / KMeans 會表現不錯
若資料呈現彎曲、非線性 → Autoencoder embedding 會更適合
(4) 是否有「維度塌縮」問題？ 如果 2D PCA 看起來像一條線：

代表資料高度相關
clustering 可能會不穩定
Autoencoder 可能需要更小的 bottleneck
'''
X = df[all_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 用於檢查/可視化（先壓到能解釋 95% variance）
pca_full = PCA(n_components=0.95, random_state=42)
Xp = pca_full.fit_transform(X_scaled)
logger.info("PCA shape: %s", Xp.shape)

# 再壓到 2 維做視覺化用
pca_2d = PCA(n_components=2, random_state=42)
Xp_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(7,6))
plt.title("PCA 2D (no clustering yet)")
sns.scatterplot(
    x=Xp_2d[:, 0], y=Xp_2d[:, 1], s=8, color='gray' #灰色點不會干擾視覺
)
plt.xlabel("PC1")
plt.ylabel("PC2")
# plt.show()
plt.savefig(f"./{output_dir}/pca_2d_no_clustering.png", dpi=300, bbox_inches='tight')
plt.close()

# ==================================
# ====== 建立 Autoencoder 模型 ======
# input_dim = X_scaled.shape[1]
# encoding_dim = 16

# input_layer = layers.Input(shape=(input_dim,))

# # Encoder
# x = layers.Dense(64, activation='relu')(input_layer)
# x = layers.Dense(32, activation='relu')(x)
# bottleneck = layers.Dense(encoding_dim, activation='linear', name='bottleneck')(x)

# # Decoder
# x = layers.Dense(32, activation='relu')(bottleneck)
# x = layers.Dense(64, activation='relu')(x)
# output_layer = layers.Dense(input_dim, activation='linear')(x)

# autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
# encoder = models.Model(inputs=input_layer, outputs=bottleneck)

# autoencoder.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
# autoencoder.summary()
# ==================================

# ==================================
# ====== 建立 Autoencoder 模型 加入 L2 正則化 + Dropout ======
input_dim = X_scaled.shape[1]
encoding_dim = 16

# ===== Autoencoder 結構 =====
input_layer = layers.Input(shape=(input_dim,))

# ===== Encoder =====
x = layers.Dense(
    64, activation='relu',
    kernel_regularizer=l2(1e-4)
)(input_layer)
x = layers.Dropout(0.2)(x)

x = layers.Dense(
    32, activation='relu',
    kernel_regularizer=l2(1e-4)
)(x)
x = layers.Dropout(0.2)(x)

bottleneck = layers.Dense(
    encoding_dim, activation='linear',
    name='bottleneck'
)(x)

# ===== Decoder =====
x = layers.Dense(
    32, activation='relu',
    kernel_regularizer=l2(5e-5)
)(bottleneck)
# Decoder dropout 降低
x = layers.Dropout(0.1)(x)

x = layers.Dense(
    64, activation='relu',
    kernel_regularizer=l2(5e-5)
)(x)
# 最後一層通常不加 Dropout

output_layer = layers.Dense(input_dim, activation='linear')(x)

# ===== Models =====
autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
encoder = models.Model(inputs=input_layer, outputs=bottleneck)

autoencoder.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')

autoencoder.summary()
# ==================================

# 訓練 Autoencoder
logger.info("\n===== Training Autoencoder =====")
history = autoencoder.fit(
    X_scaled, X_scaled,
    epochs=200,
    batch_size=256,
    shuffle=True,
    validation_split=0.1,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ],
    verbose=0
)
# 視覺化訓練過程
plt.figure(figsize=(10, 5))

plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)

plt.title("Autoencoder Training Curve", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("MSE Loss", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
# plt.show()
plt.savefig(f"./{output_dir}/autoencoder_training_curve.png", dpi=300, bbox_inches='tight')
plt.close()
timer.log("Autoencoder training")

# 取出 Autoencoder embedding
X_emb = encoder.predict(X_scaled)
logger.info("Embedding shape: %s", X_emb.shape)

# ====== PCA 2D for visualization ======
pca_emb_2d = PCA(n_components=2, random_state=42)
X_emb_2d = pca_emb_2d.fit_transform(X_emb)



# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 輸入想比較的 k 值
k_list = [8, 10, 12,14,16]

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



logger.info("\n===== 分群品質指標比較 =====")

# 每個 k 輸出一張獨立圖片
for k in k_list:

    # ===== KMeans 分群 ======
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels_np = kmeans.fit_predict(X_emb)

    # ===== 計算分群品質指標 ======
    sil = silhouette_score(X_emb, labels_np)
    ch = calinski_harabasz_score(X_emb, labels_np)
    dbi = davies_bouldin_score(X_emb, labels_np)

    logger.info("\n--- k = %d ---", k)
    logger.info("Silhouette Score: %f", sil)
    logger.info("Calinski-Harabasz Score: %f", ch)
    logger.info("Davies-Bouldin Index: %f", dbi)

    # ===== 每個 k 建立一張新的圖 =====
    plt.figure(figsize=(7, 6))

    sns.scatterplot(
        x=X_emb_2d[:, 0],
        y=X_emb_2d[:, 1],
        hue=labels_np,
        palette="Paired",
        s=10,
        linewidth=0
    )

    plt.title(f"KMeans Clustering (k={k})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    # 顯示 legend
    plt.legend(title=f"k={k}", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # ===== 輸出獨立圖片 =====
    plt.savefig(f"{output_dir}/kmeans_clustering_k{k}.png",
                dpi=300, bbox_inches='tight')

    plt.close()

# 輸出每個 k 的 KMeans labels
for k in k_list:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels_np = kmeans.fit_predict(X_emb)

    # 建立欄位名稱，例如 cluster_ae_kmeans_k8
    col_name = f"cluster_ae_kmeans_k{k}"
    df[col_name] = labels_np

    print(f"已套用 k = {k} 的分群結果，欄位名稱：{col_name}")

# 每個 k 建立一張新的圖
for k in k_list:
    cluster_col = f"cluster_ae_kmeans_k{k}"   # 對應欄位名稱
    plt.figure(figsize=(7, 6))

    sns.scatterplot(
        x=Xp_2d[:, 0],
        y=Xp_2d[:, 1],
        hue=df[cluster_col],
        palette="Paired",
        s=10,
        linewidth=0
    )

    plt.title(f"Original PCA 2D (k={k})")
    plt.xlabel("Original PC1")
    plt.ylabel("Original PC2")

    # 顯示 legend（如果你想關掉可改成空 legend）
    plt.legend(title=f"k={k}", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    # plt.show()
    # 每個 k 輸出一張獨立圖片
    plt.savefig(f"{output_dir}/original_pca_2d_k{k}.png",
                dpi=300, bbox_inches='tight')

    plt.close()

timer.log("KMeans clustering")


"""
自動命名每個 cluster，保證名稱唯一且可讀
"""
def auto_name_clusters_by_k(df, cluster_col='cluster_ae_kmeans', semantic_features=None,name_col="cluster_name"):
    if semantic_features is None:
        semantic_features = [
            'energy', 'danceability', 'valence', 'acousticness',
            'instrumentalness', 'tempo', 'loudness',
            'speechiness', 'liveness'
        ]

    # 計算每個 cluster 的中心點
    centroids = df.groupby(cluster_col)[semantic_features].mean()

    # 建立 quantile 區間（每個 feature 會有低、中、高三段）
    quantiles = df[semantic_features].quantile([0.33, 0.66])
    q33 = quantiles.loc[0.33]
    q66 = quantiles.loc[0.66]

    cluster_names = {}
    used_names = set()

    for cluster_id, row in centroids.iterrows():
        descriptors = []

        for feature in semantic_features:
            # 大寫開頭特徵名稱（energy → Energy）
            feat_name = feature.capitalize()

            # 取該特徵的 quantile 切點
            f33 = q33[feature]
            f66 = q66[feature]

            # 低
            if row[feature] < f33:
                descriptors.append(f"Low {feat_name}")
            # 高
            elif row[feature] > f66:
                descriptors.append(f"High {feat_name}")
            # 中間則不命名（避免太冗長）
            else:
                continue

        # 若全部都是中間數值 → 給預設描述
        if not descriptors:
            descriptors.append("Balanced")

        # 合併名稱
        name = " / ".join(descriptors)

        # 避免名稱重複
        if name in used_names:
            name = f"{name} (Cluster {cluster_id})"
        used_names.add(name)

        cluster_names[cluster_id] = name

    # df['cluster_name'] = df[cluster_col].map(cluster_names)
    # df[name_col] = cluster_names[df[cluster_col]]
    df[name_col] = df[cluster_col].map(cluster_names)


    return df, cluster_names

# 視覺化自動命名後的結果
semantic_features = [
            'energy', 'danceability', 'valence', 'acousticness',
            'instrumentalness', 'tempo', 'loudness',
            'speechiness', 'liveness'
        ]

for k in k_list:

    cluster_col = f"cluster_ae_kmeans_k{k}"
    name_col = f"cluster_name_k{k}"

    # 自動命名 cluster
    df, cluster_names = auto_name_clusters_by_k(
        df,
        cluster_col=cluster_col,
        semantic_features=semantic_features,
        name_col=name_col
    )

    # 每個 k 建立一張新的圖
    plt.figure(figsize=(7, 6))

    sns.scatterplot(
        x=X_emb_2d[:, 0],
        y=X_emb_2d[:, 1],
        hue=df[name_col],
        palette="Paired",
        s=10,
        linewidth=0
    )

    plt.title(f"KMeans on AE Embedding (k={k})")
    plt.xlabel("Embedding PC1")
    plt.ylabel("Embedding PC2")

    # 顯示 cluster 名稱
    plt.legend(title=f"k={k}", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # 每個 k 輸出一張獨立圖片
    plt.savefig(f"{output_dir}/kmeans_on_ae_embedding_k{k}.png",
                dpi=300, bbox_inches='tight')

    plt.close()

def recommend_song_ae_cluster(cluster_col,embedding_cols,song_name,artist=None,n_recommendations=5,max_per_cluster=2):
    """
    Autoencoder + Cluster + Cosine Similarity 推薦系統
    - cluster_col: 使用哪個 cluster 欄位（例如 'cluster_ae_kmeans'）
    - embedding_cols: Autoencoder embedding 欄位（list）
    - max_per_cluster: 跨 cluster 時，每個 cluster 最多取幾首
    """

    # 找目標歌曲
    if artist:
        song = df[(df['track_name'].str.lower() == song_name.lower()) &
                  (df['artists'].str.lower() == artist.lower())]
    else:
        song = df[df['track_name'].str.lower() == song_name.lower()]

    if song.empty:
        logger.info("❌ Song '%s' not found in dataset.", song_name)
        return None

    song_index = song.index[0]
    song_emb = df.loc[song_index, embedding_cols].values.reshape(1, -1)
    song_cluster = df.loc[song_index, cluster_col]

    # 同 cluster 篩選
    same_cluster = df[(df[cluster_col] == song_cluster) & (df.index != song_index)].copy()

    # 計算 cosine similarity
    same_cluster['distance'] = cosine_similarity(
        same_cluster[embedding_cols].values, song_emb
    ).reshape(-1)

    # 先取同 cluster 的推薦
    recommendations = same_cluster.sort_values('distance', ascending=False).head(n_recommendations)

    # fallback：跨 cluster，但限制每個 cluster 最多 max_per_cluster 首
    if len(recommendations) < n_recommendations:

        remaining_n = n_recommendations - len(recommendations)

        other = df[df.index != song_index].copy()
        other = other[~other.index.isin(recommendations.index)]

        # 計算 cosine similarity
        other['distance'] = cosine_similarity(
            other[embedding_cols].values, song_emb
        ).reshape(-1)

        # 按 cluster 分組，每個 cluster 取 max_per_cluster 首
        fallback_list = []
        for c, group in other.groupby(cluster_col):
            top_c = group.sort_values('distance', ascending=False).head(max_per_cluster)
            fallback_list.append(top_c)

        fallback_df = pd.concat(fallback_list).sort_values('distance', ascending=False)

        # 取剩下需要的數量
        fallback_final = fallback_df.head(remaining_n)

        # 合併
        recommendations = pd.concat([recommendations, fallback_final])

    # 輸出格式統一
    recommendations = recommendations.copy()
    recommendations['cluster'] = recommendations[cluster_col]
    recommendations = recommendations[['track_name', 'artists', 'cluster', 'distance']]

    logger.info("\n🎵 Recommendations for '%s' (Artist: %s):", song_name, artist if artist else 'Any')
    # print(recommendations)

    return recommendations

# 自動建立 embedding 欄位名稱
embedding_cols = [f"emb_{i}" for i in range(X_emb.shape[1])]
# 寫入 df
df[embedding_cols] = X_emb
# recommend_song_ae_cluster(cluster_col="cluster_ae_kmeans_k8",embedding_cols=embedding_cols,song_name="Comedy",artist="Gen Hoshino")

for k in k_list:
    cluster_col = f"cluster_ae_kmeans_k{k}"
    logger.info("\n===== 推薦結果（k = %d）=====", k)

    rec = recommend_song_ae_cluster(
        cluster_col=cluster_col,
        embedding_cols=embedding_cols,
        song_name="Comedy",
        artist="Gen Hoshino"
    )

    logger.info("\n%s", rec)
timer.log("Recommendation")

# ====== 記錄整體結束時間 ======
experiment_end = time.time()
total_elapsed = experiment_end - experiment_start

h = int(total_elapsed // 3600)
m = int((total_elapsed % 3600) // 60)
s = total_elapsed % 60

logger.info("===== Experiment finished =====")
logger.info("Total runtime: %d h %d m %.2f s", h, m, s)
