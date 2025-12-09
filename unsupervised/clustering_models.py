"""
Módulo de modelos de clustering não-supervisionado.
Tech Challenge Fase 3 - Modelo Não-Supervisionado
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# import seaborn as sns


class ClusteringAnalyzer:
    """Classe para análise de clustering com múltiplos algoritmos."""
    
    def __init__(self):
        """Inicializa o analisador de clustering."""
        self.results = {}
        self.best_model = None
        
    def kmeans_clustering(
        self,
        X: np.ndarray,
        n_clusters: int = 5,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Aplica K-Means clustering.
        
        Args:
            X: Dados normalizados
            n_clusters: Número de clusters
            random_state: Seed para reprodutibilidade
            
        Returns:
            Dicionário com resultados do clustering
        """
        print(f"\n=== K-Means Clustering (k={n_clusters}) ===")
        
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
        
        labels = model.fit_predict(X)
        
        # Calcular métricas
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        davies = davies_bouldin_score(X, labels)
        
        result = {
            'algorithm': 'KMeans',
            'model': model,
            'labels': labels,
            'centers': model.cluster_centers_,
            'n_clusters': n_clusters,
            'inertia': model.inertia_,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'davies_bouldin_score': davies,
            'cluster_sizes': np.bincount(labels)
        }
        
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Calinski-Harabasz Score: {calinski:.2f}")
        print(f"Davies-Bouldin Score: {davies:.4f}")
        print(f"Inertia: {model.inertia_:.2f}")
        print(f"Tamanhos dos clusters: {dict(enumerate(result['cluster_sizes']))}")
        
        self.results['kmeans'] = result
        return result
    
    def dbscan_clustering(
        self,
        X: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 5
    ) -> Dict[str, Any]:
        """
        Aplica DBSCAN clustering.
        
        Args:
            X: Dados normalizados
            eps: Distância máxima entre amostras
            min_samples: Número mínimo de amostras em um cluster
            
        Returns:
            Dicionário com resultados do clustering
        """
        print(f"\n=== DBSCAN Clustering (eps={eps}, min_samples={min_samples}) ===")
        
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        
        # Número de clusters (excluindo ruído -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        result = {
            'algorithm': 'DBSCAN',
            'model': model,
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'eps': eps,
            'min_samples': min_samples
        }
        
        # Calcular métricas apenas se houver mais de 1 cluster
        if n_clusters > 1 and n_noise < len(labels):
            # Remover pontos de ruído para cálculo de métricas
            mask = labels != -1
            if mask.sum() > 0:
                silhouette = silhouette_score(X[mask], labels[mask])
                calinski = calinski_harabasz_score(X[mask], labels[mask])
                davies = davies_bouldin_score(X[mask], labels[mask])
                
                result['silhouette_score'] = silhouette
                result['calinski_harabasz_score'] = calinski
                result['davies_bouldin_score'] = davies
                
                print(f"Silhouette Score: {silhouette:.4f}")
                print(f"Calinski-Harabasz Score: {calinski:.2f}")
                print(f"Davies-Bouldin Score: {davies:.4f}")
        
        print(f"Número de clusters: {n_clusters}")
        print(f"Pontos de ruído: {n_noise}")
        
        unique, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(unique) > 0:
            print(f"Tamanhos dos clusters: {dict(zip(unique, counts))}")
        
        self.results['dbscan'] = result
        return result
    
    
    def find_optimal_k(
        self,
        X: np.ndarray,
        k_range: range = range(2, 11)
    ) -> pd.DataFrame:
        """
        Encontra o número ótimo de clusters testando diferentes valores de k.
        
        Args:
            X: Dados normalizados
            k_range: Range de valores de k para testar
            
        Returns:
            DataFrame com métricas para cada k
        """
        print("\n=== Buscando k ótimo para K-MEANS ===")
        
        results = []
        
        for k in k_range:
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X)
            
            silhouette = silhouette_score(X, labels)
            calinski = calinski_harabasz_score(X, labels)
            davies = davies_bouldin_score(X, labels)
            
            result = {
                'k': k,
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski,
                'davies_bouldin_score': davies,
                'inertia': model.inertia_
            }
            
            results.append(result)
            print(f"k={k}: Silhouette={silhouette:.4f}, Calinski={calinski:.2f}, Davies-Bouldin={davies:.4f}")
        
        df_results = pd.DataFrame(results)
        
        # Encontrar melhor k baseado em silhouette score
        best_k = df_results.loc[df_results['silhouette_score'].idxmax(), 'k']
        print(f"\nMelhor k (baseado em Silhouette): {int(best_k)}")
        
        return df_results
    
    def visualize_clusters_2d(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        title: str = "Clustering Results",
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualiza clusters em 2D usando PCA.
        
        Args:
            X: Dados originais
            labels: Labels dos clusters
            title: Título do gráfico
            save_path: Caminho para salvar o gráfico (opcional)
        """
        # Reduzir para 2D com PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        # Criar gráfico
        plt.figure(figsize=(12, 8))
        
        # Plotar pontos
        scatter = plt.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=labels,
            cmap='viridis',
            alpha=0.6,
            edgecolors='w',
            linewidth=0.5
        )
        
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variância)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variância)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo em: {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def analyze_clusters(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        feature_columns: List[str]
    ) -> pd.DataFrame:
        """
        Analisa características de cada cluster.
        
        Args:
            df: DataFrame original com os dados
            labels: Labels dos clusters
            feature_columns: Colunas de features para analisar
            
        Returns:
            DataFrame com estatísticas por cluster
        """
        df_analysis = df.copy()
        df_analysis['cluster'] = labels

        return df_analysis.groupby('cluster')[feature_columns].agg(
            ['mean', 'std', 'count']
        )
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compara todos os modelos executados.
        
        Returns:
            DataFrame com comparação de métricas
        """
        if not self.results:
            print("Nenhum modelo foi executado ainda.")
            return pd.DataFrame()
        
        comparison = []
        
        for name, result in self.results.items():
            row = {
                'algorithm': name,
                'n_clusters': result.get('n_clusters', 'N/A')
            }
            
            if 'silhouette_score' in result:
                row['silhouette_score'] = result['silhouette_score']
            if 'calinski_harabasz_score' in result:
                row['calinski_harabasz_score'] = result['calinski_harabasz_score']
            if 'davies_bouldin_score' in result:
                row['davies_bouldin_score'] = result['davies_bouldin_score']
            if 'inertia' in result:
                row['inertia'] = result['inertia']
            
            comparison.append(row)
        
        df_comparison = pd.DataFrame(comparison)
        
        print("\n=== Comparação de Modelos ===")
        print(df_comparison.to_string(index=False))
        
        return df_comparison
    
    def get_best_model(self) -> Dict[str, Any]:
        """
        Retorna o melhor modelo baseado no silhouette score.
        
        Returns:
            Dicionário com informações do melhor modelo
        """
        if not self.results:
            return {}
        
        best_name = None
        best_score = -1
        
        for name, result in self.results.items():
            if 'silhouette_score' in result and result['silhouette_score'] > best_score:
                best_score = result['silhouette_score']
                best_name = name
        
        if best_name:
            print(f"\nMelhor modelo: {best_name.upper()} "
                f"(Silhouette Score: {best_score:.4f})")
            return self.results[best_name]
        
        return {}

