# Tech Challenge Fase 3 - Modelo Não-Supervisionado
## Machine Learning Engineering

Este projeto implementa modelos de **clustering não-supervisionado** para análise de dados de voos, aeroportos e companhias aéreas.

---

## Descrição do Projeto

O objetivo é aplicar algoritmos de aprendizado não-supervisionado para identificar padrões e agrupamentos nos dados de voos domésticos dos EUA em 2015, incluindo:

- **Clustering de Aeroportos**: Identificar grupos de aeroportos com características operacionais similares
- **Clustering de Companhias Aéreas**: Agrupar companhias por padrões de pontualidade e eficiência
- **Clustering de Rotas**: Identificar rotas com comportamentos similares

---

## Tecnologias Utilizadas

- **Python 3.8+**
- **scikit-learn**: Algoritmos de clustering (KMeans, DBSCAN)
- **pandas**: Manipulação e análise de dados
- **numpy**: Computação numérica
- **matplotlib & seaborn**: Visualização de dados
- **jupyter**: Notebooks interativos

---

## Estrutura do Projeto

```
unsupervised/
├── data_loader.py              
├── clustering_models.py        
├── main.py                     
├── analise_exploratoria.ipynb  
├── requirements.txt            
├── README.md                   
└── results/                    
    ├── clustering_airport.csv
    ├── clustering_airline.csv
    ├── model_comparison_*.csv
    └── cluster_analysis_*.csv
```

---

## Como Executar

### 1. Instalação das Dependências

```bash
pip install -r requirements.txt
```

### 2. Execução via Script Principal

#### Clustering de Aeroportos (padrão)
```bash
python main.py --sample 100000 --entity airport --k 5
```

#### Clustering de Companhias Aéreas
```bash
python main.py --sample 100000 --entity airline --k 4
```

#### Clustering de Rotas
```bash
python main.py --sample 100000 --entity route --k 6
```

#### Buscar número ótimo de clusters
```bash
python main.py --sample 100000 --entity airport --find-k
```

#### Usar todos os dados (sem amostragem)
```bash
python main.py --sample 0 --entity airport --k 5
```

### 3. Parâmetros Disponíveis

- `--sample`: Número de registros para amostrar (default: 100000, use 0 para todos)
- `--entity`: Entidade para clusterizar (`airport`, `airline`, `route`)
- `--k`: Número de clusters (default: 5)
- `--find-k`: Flag para buscar o k ótimo automaticamente

### 4. Execução via Jupyter Notebook

```bash
jupyter notebook analise_exploratoria.ipynb
```

O notebook contém:
- Análise exploratória completa
- Visualizações interativas
- Comparação de algoritmos
- Interpretação dos resultados

---

## Algoritmos Implementados

### 1. **K-Means**
- Algoritmo de particionamento baseado em centróides
- Agrupa dados em k clusters predefinidos
- Melhor para clusters esféricos e de tamanho similar

### 2. **DBSCAN** (Density-Based Spatial Clustering)
- Baseado em densidade
- Identifica clusters de formas arbitrárias
- Detecta outliers (pontos de ruído)


---

## Métricas de Avaliação

### Silhouette Score
- Varia de -1 a 1
- **Maior é melhor**
- Mede a qualidade dos clusters

### Calinski-Harabasz Score
- Quanto maior, melhor
- Razão entre dispersão inter-cluster e intra-cluster

### Davies-Bouldin Score
- **Menor é melhor**
- Mede a separação entre clusters

### Inertia (apenas K-Means)
- Soma das distâncias quadradas ao centróide mais próximo
- Usado no método do cotovelo (elbow method)

---

## 📂 Arquivos de Saída

Todos os resultados são salvos na pasta `results/`:

1. **clustering_{entity}.csv**: Dados originais com labels de clusters
2. **model_comparison_{entity}.csv**: Comparação de métricas entre algoritmos
3. **cluster_analysis_{entity}.csv**: Estatísticas descritivas por cluster
4. **optimal_k_{entity}.csv**: Métricas para diferentes valores de k

---

## Interpretação dos Resultados

### Aeroportos
Os clusters podem representar:
- **Hubs principais** vs **aeroportos regionais**
- Níveis de **congestionamento** e **atrasos**
- Padrões **geográficos** de operação
- Diferentes níveis de **eficiência operacional**

### Companhias Aéreas
Os clusters podem indicar:
- Companhias **tradicionais** vs **low-cost**
- Diferentes níveis de **pontualidade**
- **Especializações** operacionais (voos curtos vs longos)
- **Qualidade** do serviço

### Rotas
Os clusters podem revelar:
- Rotas **populares** vs **secundárias**
- Níveis de **competitividade** (múltiplas companhias)
- Rotas com **maiores atrasos**
- Padrões de **distância** e **duração**

---

## Exemplos de Uso

### Exemplo 1: Análise Rápida de Aeroportos
```python
from data_loader import DataLoader
from clustering_models import ClusteringAnalyzer

# Carregar dados
loader = DataLoader(base_path=".")
loader.load_data(sample_size=50000)

# Processar aeroportos
df_airports = loader.get_airport_statistics()
X, df_clean = loader.prepare_features_for_clustering(
    df_airports,
    ['total_flights', 'avg_arrival_delay', 'cancellation_rate']
)

# Aplicar K-Means
analyzer = ClusteringAnalyzer()
result = analyzer.kmeans_clustering(X, n_clusters=5)

# Visualizar
analyzer.visualize_clusters_2d(X, result['labels'], title="Aeroportos - K-Means")
```

### Exemplo 2: Comparar Múltiplos Algoritmos
```python
# K-Means
kmeans_result = analyzer.kmeans_clustering(X, n_clusters=5)

# DBSCAN
dbscan_result = analyzer.dbscan_clustering(X, eps=0.5, min_samples=5)


# Comparar
comparison = analyzer.compare_models()
print(comparison)

# Melhor modelo
best = analyzer.get_best_model()
print(f"Melhor: {best['algorithm']}")
```

---

## Features Utilizadas

### Aeroportos
- Volume de voos
- Atrasos médios (partida e chegada)
- Desvio padrão dos atrasos
- Distância média dos voos
- Taxa de cancelamento
- Taxa de desvio
- Tempo médio de voo
- Localização geográfica (latitude/longitude)

### Companhias Aéreas
- Volume de voos
- Atrasos médios
- Distância média
- Taxas de cancelamento e desvio
- Tempo de taxiamento
- Tempo de voo

### Rotas
- Frequência de voos
- Atrasos médios
- Distância
- Taxa de cancelamento
- Número de companhias operando
- Tempo de voo

---

## Notas Técnicas

### Pré-processamento
- Valores faltantes são preenchidos com a mediana
- Features são normalizadas usando StandardScaler
- Aeroportos/rotas com poucos voos são filtrados

### Performance
- Para análise rápida, use `--sample 100000`
- Para análise completa, use `--sample 0` (pode demorar)
- DBSCAN pode ser lento em datasets grandes

### Visualização
- PCA é usado para reduzir dimensionalidade para 2D
- Gráficos são salvos automaticamente em alta resolução
- Use o notebook para visualizações interativas

---

## 🤝 Contribuindo

Este projeto foi desenvolvido para o Tech Challenge Fase 3 da pós-graduação em Machine Learning Engineering.

---

## 📧 Autor

Desenvolvido para o Tech Challenge - FIAP/Alura
Machine Learning Engineering - Fase 3

---

## 📄 Licença

Este projeto é parte de um trabalho acadêmico.

---

## Referências

- [Documentação scikit-learn](https://scikit-learn.org/)
- [K-Means Clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan)

---

## Checklist de Execução

- [ ] Instalar dependências (`pip install -r requirements.txt`)
- [ ] Verificar que os arquivos CSV estão na pasta pai (`../`)
- [ ] Executar análise exploratória (`python main.py --entity airport`)
- [ ] Buscar k ótimo (`python main.py --find-k`)
- [ ] Executar clustering com k ótimo
- [ ] Analisar resultados na pasta `results/`
- [ ] Explorar visualizações no notebook
- [ ] Interpretar os clusters identificados
- [ ] Documentar insights encontrados

---

**Boa análise!**

