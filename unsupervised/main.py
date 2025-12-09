"""
Tech Challenge Fase 3 - Machine Learning Engineering

(KMeans e DBSCAN).
"""

# import sys
# import pandas as pd
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from data_loader import DataLoader
from clustering_models import ClusteringAnalyzer


def main():
    """Função principal de execução."""
    parser = argparse.ArgumentParser(
        description='Modelo Não-Supervisionado - Tech Challenge Fase 3'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=100000,
        help='Número de registros para amostrar (default: 100.000, use 0 para todos)'
    )
    parser.add_argument(
        '--entity',
        type=str,
        choices=['airport', 'airline', 'route'],
        default='airport',
        help='Entidade para clusterizar (airport, airline, route)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Número de clusters para KMeans (default: 5)'
    )
    parser.add_argument(
        '--find-k',
        action='store_true',
        help='Buscar o N para o K-Means'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Tech Challenge Fase 3 - Modelo Não-Supervisionado")
    print("Machine Learning Engineering")
    print("=" * 70)
    
    # ===== 1. CARREGAR DADOS =====
    data_loader = DataLoader(base_path="./")
    
    sample_size = None if args.sample == 0 else args.sample
    data_loader.load_data(sample_size=sample_size)
    
    # Mostrar resumo
    print("\n" + "=" * 70)
    print("RESUMO DOS DADOS")
    print("=" * 70)
    summary = data_loader.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # ===== 2. PROCESSAR DADOS POR ENTIDADE =====
    print("\n" + "=" * 70)
    print(f"PROCESSAMENTO - {args.entity.upper()}")
    print("=" * 70)
    
    if args.entity == 'airport':
        df_stats = data_loader.get_airport_statistics()
        feature_columns = [
            'total_flights', 'avg_departure_delay', 'std_departure_delay',
            'avg_arrival_delay', 'std_arrival_delay', 'avg_distance',
            'cancellation_rate', 'diversion_rate', 'avg_air_time',
            'LATITUDE', 'LONGITUDE'
        ]
        entity_column = 'AIRPORT'
        
    elif args.entity == 'airline':
        df_stats = data_loader.get_airline_statistics()
        feature_columns = [
            'total_flights', 'avg_departure_delay', 'std_departure_delay',
            'avg_arrival_delay', 'std_arrival_delay', 'avg_distance',
            'cancellation_rate', 'diversion_rate', 'avg_air_time',
            'avg_taxi_out', 'avg_taxi_in'
        ]
        entity_column = 'AIRLINE'
        
    else:  # route
        df_stats = data_loader.get_route_statistics()
        feature_columns = [
            'total_flights', 'avg_departure_delay', 'std_departure_delay',
            'avg_arrival_delay', 'std_arrival_delay', 'distance',
            'cancellation_rate', 'avg_air_time', 'num_airlines'
        ]
        entity_column = 'ORIGIN'
    
    # Preparar features
    X, df_clean = data_loader.prepare_features_for_clustering(
        df_stats,
        feature_columns
    )
    
    # ===== 3. BUSCAR K ÓTIMO (SE SOLICITADO) =====
    if args.find_k:
        print("\n" + "=" * 70)
        print("BUSCA DO K ÓTIMO")
        print("=" * 70)
        
        analyzer = ClusteringAnalyzer()
        df_k_results = analyzer.find_optimal_k(X, k_range=range(2, 11))
        
        # Salvar resultados
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        df_k_results.to_csv(f"{output_dir}/optimal_k_{args.entity}.csv", index=False)
        print(f"\nResultados salvos em: {output_dir}/optimal_k_{args.entity}.csv")
    
    # ===== 4. EXECUTAR ALGORITMOS DE CLUSTERING =====
    print("\n" + "=" * 70)
    print("EXECUÇÃO DOS ALGORITMOS DE CLUSTERING")
    print("=" * 70)
    
    analyzer = ClusteringAnalyzer()
    
    # K-Means
    kmeans_result = analyzer.kmeans_clustering(X, n_clusters=args.k)
    df_clean['cluster_kmeans'] = kmeans_result['labels']
    
    # DBSCAN (ajustar eps baseado no tamanho dos dados)
    eps_value = 0.5 if len(X) < 1000 else 0.3
    dbscan_result = analyzer.dbscan_clustering(X, eps=eps_value, min_samples=5)
    df_clean['cluster_dbscan'] = dbscan_result['labels']
    
    # ===== 5. COMPARAR MODELOS =====
    print("\n" + "=" * 70)
    print("COMPARAÇÃO DOS MODELOS")
    print("=" * 70)
    
    df_comparison = analyzer.compare_models()
    best_model = analyzer.get_best_model()
    
    # ===== 6. ANALISAR CLUSTERS DO MELHOR MODELO =====
    print("\n" + "=" * 70)
    print("ANÁLISE DOS CLUSTERS (MELHOR MODELO)")
    print("=" * 70)
    
    best_labels = best_model['labels']
    df_clean['best_cluster'] = best_labels
    
    cluster_analysis = analyzer.analyze_clusters(
        df_clean,
        best_labels,
        feature_columns
    )
    
    print("\nEstatísticas por Cluster:")
    print(cluster_analysis)
    
    # ===== 7. SALVAR RESULTADOS =====
    print("\n" + "=" * 70)
    print("SALVANDO RESULTADOS")
    print("=" * 70)
    
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar dados com clusters
    output_file = f"{output_dir}/clustering_{args.entity}.csv"
    df_clean.to_csv(output_file, index=False)
    print(f"Dados com clusters salvos em: {output_file}")
    
    # Salvar comparação de modelos
    comparison_file = f"{output_dir}/model_comparison_{args.entity}.csv"
    df_comparison.to_csv(comparison_file, index=False)
    print(f"Comparação de modelos salva em: {comparison_file}")
    
    # Salvar análise de clusters
    analysis_file = f"{output_dir}/cluster_analysis_{args.entity}.csv"
    cluster_analysis.to_csv(analysis_file)
    print(f"Análise de clusters salva em: {analysis_file}")
    
    # ===== 8. VISUALIZAÇÕES =====
    print("\n" + "=" * 70)
    print("GERANDO VISUALIZAÇÕES")
    print("=" * 70)
    
    try:
        # Configurar estilo
        sns.set_palette("husl")
        
        # 1. Clusters em 2D (PCA)
        print("\n1. Criando visualização dos clusters em 2D...")
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # K-Means
        scatter1 = axes[0].scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=kmeans_result['labels'],
            cmap='viridis',
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5,
            s=50
        )
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variância)', fontsize=11)
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variância)', fontsize=11)
        axes[0].set_title(f'K-Means Clustering (k={args.k})', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0], label='Cluster')
        
        # DBSCAN
        scatter2 = axes[1].scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=dbscan_result['labels'],
            cmap='viridis',
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5,
            s=50
        )
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variância)', fontsize=11)
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variância)', fontsize=11)
        axes[1].set_title('DBSCAN Clustering', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[1], label='Cluster (-1 = ruído)')
        
        plt.tight_layout()
        plot1_file = f"{output_dir}/clusters_2d_{args.entity}.png"
        plt.savefig(plot1_file, dpi=300, bbox_inches='tight')
        print(f"   Salvo: {plot1_file}")
        plt.close()
        
        # 2. Comparação de Métricas
        print("\n2. Criando comparação de métricas...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
        titles = ['Silhouette Score\n(maior é melhor)', 
                  'Calinski-Harabasz\n(maior é melhor)', 
                  'Davies-Bouldin\n(menor é melhor)']
        colors = ['green', 'blue', 'red']
        
        for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
            if metric in df_comparison.columns:
                df_plot = df_comparison.dropna(subset=[metric])
                axes[i].bar(df_plot['algorithm'], df_plot[metric], color=color, alpha=0.7)
                axes[i].set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
                axes[i].set_title(title, fontsize=12, fontweight='bold')
                axes[i].tick_params(axis='x', rotation=0)
                axes[i].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot2_file = f"{output_dir}/metrics_comparison_{args.entity}.png"
        plt.savefig(plot2_file, dpi=300, bbox_inches='tight')
        print(f"   Salvo: {plot2_file}")
        plt.close()
        
        # 3. Características dos Clusters
        print("\n3. Criando análise de características dos clusters...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Selecionar features importantes para visualização
        if args.entity == 'airport':
            viz_features = {
                'total_flights': ('Média de Voos', 'steelblue'),
                'avg_arrival_delay': ('Atraso Médio (min)', 'coral'),
                'cancellation_rate': ('Taxa de Cancelamento', 'red'),
                'avg_distance': ('Distância Média (milhas)', 'green')
            }
        elif args.entity == 'airline':
            viz_features = {
                'total_flights': ('Média de Voos', 'steelblue'),
                'avg_arrival_delay': ('Atraso Médio (min)', 'coral'),
                'cancellation_rate': ('Taxa de Cancelamento', 'red'),
                'avg_taxi_out': ('Tempo Taxi Out (min)', 'green')
            }
        else:  # route
            viz_features = {
                'total_flights': ('Frequência de Voos', 'steelblue'),
                'avg_arrival_delay': ('Atraso Médio (min)', 'coral'),
                'cancellation_rate': ('Taxa de Cancelamento', 'red'),
                'num_airlines': ('Nº de Companhias', 'green')
            }
        
        for idx, (feature, (label, color)) in enumerate(viz_features.items()):
            if feature in df_clean.columns:
                row = idx // 2
                col = idx % 2
                df_clean.groupby('best_cluster')[feature].mean().plot(
                    kind='bar', ax=axes[row, col], color=color, alpha=0.7
                )
                axes[row, col].set_title(f'{label} por Cluster', fontsize=12, fontweight='bold')
                axes[row, col].set_ylabel(label, fontsize=10)
                axes[row, col].set_xlabel('Cluster', fontsize=10)
                axes[row, col].tick_params(axis='x', rotation=0)
                axes[row, col].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot3_file = f"{output_dir}/cluster_characteristics_{args.entity}.png"
        plt.savefig(plot3_file, dpi=300, bbox_inches='tight')
        print(f"   Salvo: {plot3_file}")
        plt.close()
        
        # 4. Mapa Geográfico (apenas para aeroportos)
        if args.entity == 'airport' and 'LATITUDE' in df_clean.columns:
            print("\n4. Criando mapa geográfico dos clusters...")
            plt.figure(figsize=(18, 10))
            
            scatter = plt.scatter(
                df_clean['LONGITUDE'],
                df_clean['LATITUDE'],
                c=df_clean['best_cluster'],
                s=df_clean['total_flights'] / 30,
                cmap='viridis',
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5
            )
            
            plt.colorbar(scatter, label='Cluster')
            plt.xlabel('Longitude', fontsize=12)
            plt.ylabel('Latitude', fontsize=12)
            plt.title(f'Distribuição Geográfica dos Clusters - {args.entity.title()}\n(tamanho do ponto = volume de voos)', 
                      fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot4_file = f"{output_dir}/geographic_map_{args.entity}.png"
            plt.savefig(plot4_file, dpi=300, bbox_inches='tight')
            print(f"   Salvo: {plot4_file}")
            plt.close()
        
        print("\nTodas as visualizações foram geradas com sucesso!")
        
    except Exception as e:
        print(f"\n[AVISO] Erro ao gerar visualizações: {e}")
        print("Os dados foram salvos em CSV, mas algumas visualizações podem não ter sido geradas.")
    
    # ===== 9. INSIGHTS PRINCIPAIS =====
    print("\n" + "=" * 70)
    print("INSIGHTS PRINCIPAIS")
    print("=" * 70)
    
    print(f"\n1. Algoritmo recomendado: {best_model['algorithm']}")
    print(f"   - Número de clusters: {best_model['n_clusters']}")
    print(f"   - Silhouette Score: {best_model.get('silhouette_score', 'N/A')}")
    
    print("\n2. Distribuição dos clusters:")
    cluster_sizes = np.bincount(best_labels[best_labels >= 0])
    for i, size in enumerate(cluster_sizes):
        print(f"   - Cluster {i}: {size} itens ({size/len(best_labels)*100:.1f}%)")
    
    if args.entity == 'airport':
        print("\n3. Interpretação (Aeroportos):")
        print("   Os clusters podem representar:")
        print("   - Aeroportos principais (hubs) vs regionais")
        print("   - Diferentes níveis de congestionamento")
        print("   - Padrões geográficos de operação")
        
    elif args.entity == 'airline':
        print("\n3. Interpretação (Companhias Aéreas):")
        print("   Os clusters podem representar:")
        print("   - Companhias tradicionais vs low-cost")
        print("   - Diferentes níveis de pontualidade")
        print("   - Especializações operacionais")
    
    print("\n" + "=" * 70)
    print("EXECUÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 70)
    print("\nArquivos gerados em './results/':")
    print(f"   - clustering_{args.entity}.csv (dados com clusters)")
    print(f"   - model_comparison_{args.entity}.csv (comparação de modelos)")
    print(f"   - cluster_analysis_{args.entity}.csv (estatísticas por cluster)")
    print(f"   - clusters_2d_{args.entity}.png (visualização 2D)")
    print(f"   - metrics_comparison_{args.entity}.png (comparação de métricas)")
    print(f"   - cluster_characteristics_{args.entity}.png (características dos clusters)")
    if args.entity == 'airport':
        print(f"   - geographic_map_{args.entity}.png (mapa geográfico)")


if __name__ == "__main__":
    main()

