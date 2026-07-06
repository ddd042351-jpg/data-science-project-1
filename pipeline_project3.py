import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_segmentation_pipeline():
    file_name = "Dataset for Data Analytics - Sheet1.csv"
    
    if not os.path.exists(file_name):
        print(f"Error: Could not find {file_name}")
        return
        
    df = pd.read_csv(file_name)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [col for col in numeric_cols if col.lower() not in ['orderid', 'customerid', 'is_fraud']]
    
    df = df.dropna(subset=features)
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)
    
    df['Cluster'] = clusters
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]
    
    score = silhouette_score(X_pca, clusters)
    
    print("--- Project 3 Clustering Metrics ---")
    print(f"Optimal Clusters (K): {optimal_k}")
    print(f"Silhouette Score:     {score:.4f}")
    print("------------------------------------")
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/customer_segments.csv", index=False)
    print("Success! Segmented data saved to data/customer_segments.csv")

if __name__ == "__main__":
    run_segmentation_pipeline()