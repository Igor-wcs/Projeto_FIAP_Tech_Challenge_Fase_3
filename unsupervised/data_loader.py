"""
Módulo de carregamento e pré-processamento de dados.
Tech Challenge Fase 3 - Modelo Não-Supervisionado
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """Classe responsável por carregar e processar os dados de voos."""
    
    def __init__(self, base_path: str = "./"):
        """
        Inicializa o DataLoader.
        
        Args:
            base_path: Caminho base onde estão os arquivos CSV
        """
        self.base_path = base_path
        self.flights_df = None
        self.airlines_df = None
        self.airports_df = None
        self.processed_data = None
        
    def load_data(self, sample_size: Optional[int] = None) -> None:
        """
        Carrega os dados dos arquivos CSV.
        
        Args:
            sample_size: Número de linhas para amostrar (None = todos os dados)
        """
        print("Carregando dados...")
        
        # Carregar airlines
        self.airlines_df = pd.read_csv(f"{self.base_path}airlines.csv")
        print(f"Airlines carregadas: {len(self.airlines_df)} registros")
        
        # Carregar airports
        self.airports_df = pd.read_csv(f"{self.base_path}airports.csv")
        print(f"Airports carregados: {len(self.airports_df)} registros")
        
        # Carregar flights (com amostragem se necessário)
        if sample_size:
            print(f"Amostrando {sample_size} registros de voos...")
            self.flights_df = pd.read_csv(
                f"{self.base_path}flights.csv",
                nrows=sample_size
            )
        else:
            print("Carregando todos os voos (pode demorar)...")
            self.flights_df = pd.read_csv(f"{self.base_path}flights.csv")
        
        print(f"Flights carregados: {len(self.flights_df)} registros")
        print(f"Período: {self.flights_df['YEAR'].min()}-{self.flights_df['MONTH'].min()} "
              f"a {self.flights_df['YEAR'].max()}-{self.flights_df['MONTH'].max()}")
    
    def get_airport_statistics(self) -> pd.DataFrame:
        """
        Agrega estatísticas por aeroporto para clustering.
        
        Returns:
            DataFrame com features agregadas por aeroporto
        """
        print("\nProcessando estatísticas dos aeroportos...")
        
        # Estatísticas de origem
        origin_stats = self.flights_df.groupby('ORIGIN_AIRPORT').agg({
            'FLIGHT_NUMBER': 'count',  # Total de voos
            'DEPARTURE_DELAY': ['mean', 'std', 'median'],
            'ARRIVAL_DELAY': ['mean', 'std'],
            'DISTANCE': ['mean', 'sum'],
            'CANCELLED': 'sum',
            'DIVERTED': 'sum',
            'AIR_TIME': 'mean'
        }).reset_index()
        
        origin_stats.columns = [
            'AIRPORT', 'total_flights', 
            'avg_departure_delay', 'std_departure_delay', 'median_departure_delay',
            'avg_arrival_delay', 'std_arrival_delay',
            'avg_distance', 'total_distance',
            'total_cancelled', 'total_diverted',
            'avg_air_time'
        ]
        
        # Adicionar informações geográficas
        origin_stats = origin_stats.merge(
            self.airports_df[['IATA_CODE', 'LATITUDE', 'LONGITUDE', 'CITY', 'STATE']],
            left_on='AIRPORT',
            right_on='IATA_CODE',
            how='left'
        )
        
        # Calcular taxas
        origin_stats['cancellation_rate'] = (
            origin_stats['total_cancelled'] / origin_stats['total_flights']
        )
        origin_stats['diversion_rate'] = (
            origin_stats['total_diverted'] / origin_stats['total_flights']
        )
        
        # Remover aeroportos com poucos voos
        origin_stats = origin_stats[origin_stats['total_flights'] >= 100]
        
        print(f"{len(origin_stats)} aeroportos com dados suficientes")
        
        return origin_stats
    
    def get_airline_statistics(self) -> pd.DataFrame:
        """
        Agrega estatísticas por companhia aérea para clustering.
        
        Returns:
            DataFrame com features agregadas por companhia
        """
        print("\nProcessando estatísticas das companhias aéreas...")
        
        airline_stats = self.flights_df.groupby('AIRLINE').agg({
            'FLIGHT_NUMBER': 'count',
            'DEPARTURE_DELAY': ['mean', 'std'],
            'ARRIVAL_DELAY': ['mean', 'std', 'median'],
            'DISTANCE': 'mean',
            'CANCELLED': 'sum',
            'DIVERTED': 'sum',
            'AIR_TIME': 'mean',
            'TAXI_OUT': 'mean',
            'TAXI_IN': 'mean'
        }).reset_index()
        
        airline_stats.columns = [
            'AIRLINE', 'total_flights',
            'avg_departure_delay', 'std_departure_delay',
            'avg_arrival_delay', 'std_arrival_delay', 'median_arrival_delay',
            'avg_distance', 'total_cancelled', 'total_diverted',
            'avg_air_time', 'avg_taxi_out', 'avg_taxi_in'
        ]
        
        # Adicionar nome da companhia
        airline_stats = airline_stats.merge(
            self.airlines_df,
            left_on='AIRLINE',
            right_on='IATA_CODE',
            how='left'
        )
        
        # Calcular taxas
        airline_stats['cancellation_rate'] = (
            airline_stats['total_cancelled'] / airline_stats['total_flights']
        )
        airline_stats['diversion_rate'] = (
            airline_stats['total_diverted'] / airline_stats['total_flights']
        )
        
        print(f"{len(airline_stats)} companhias processadas")
        
        return airline_stats
    
    def get_route_statistics(self) -> pd.DataFrame:
        """
        Agrega estatísticas por rota (origem-destino) para clustering.
        
        Returns:
            DataFrame com features agregadas por rota
        """
        print("\nProcessando estatísticas das rotas...")
        
        route_stats = self.flights_df.groupby(['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']).agg({
            'FLIGHT_NUMBER': 'count',
            'DEPARTURE_DELAY': ['mean', 'std'],
            'ARRIVAL_DELAY': ['mean', 'std'],
            'DISTANCE': 'mean',
            'CANCELLED': 'sum',
            'AIR_TIME': 'mean',
            'AIRLINE': 'nunique'  # Número de companhias na rota
        }).reset_index()
        
        route_stats.columns = [
            'ORIGIN', 'DESTINATION', 'total_flights',
            'avg_departure_delay', 'std_departure_delay',
            'avg_arrival_delay', 'std_arrival_delay',
            'distance', 'total_cancelled', 'avg_air_time',
            'num_airlines'
        ]
        
        # Calcular taxa de cancelamento
        route_stats['cancellation_rate'] = (
            route_stats['total_cancelled'] / route_stats['total_flights']
        )
        
        # Filtrar rotas com poucos voos
        route_stats = route_stats[route_stats['total_flights'] >= 50]
        
        print(f"{len(route_stats)} rotas com dados suficientes")
        
        return route_stats
    
    def prepare_features_for_clustering(
        self, 
        data: pd.DataFrame, 
        feature_columns: list
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Prepara features para clustering (normalização e tratamento de valores faltantes).
        
        Args:
            data: DataFrame com os dados
            feature_columns: Lista de colunas para usar como features
            
        Returns:
            Tuple com (features normalizadas, dataframe original limpo)
        """
        print(f"\nPreparando {len(feature_columns)} features para clustering...")
        
        # Criar cópia dos dados
        df_clean = data.copy()
        
        # Selecionar apenas as colunas de features
        features_df = df_clean[feature_columns].copy()
        
        # Tratar valores faltantes
        features_df = features_df.fillna(features_df.median())
        
        # Tratar infinitos
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(features_df.median())
        
        # Normalizar features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_df)
        
        print(f"Features preparadas: shape {features_normalized.shape}")
        
        return features_normalized, df_clean
    
    def get_summary(self) -> dict:
        """Retorna um resumo dos dados carregados."""
        if self.flights_df is None:
            return {"error": "Dados não carregados"}
        
        return {
            "total_flights": len(self.flights_df),
            "total_airlines": self.flights_df['AIRLINE'].nunique(),
            "total_airports": len(set(self.flights_df['ORIGIN_AIRPORT'].unique()) | 
                                  set(self.flights_df['DESTINATION_AIRPORT'].unique())),
            "date_range": f"{self.flights_df['YEAR'].min()}-{self.flights_df['MONTH'].min():02d} "
                         f"até {self.flights_df['YEAR'].max()}-{self.flights_df['MONTH'].max():02d}",
            "cancellation_rate": f"{(self.flights_df['CANCELLED'].sum() / len(self.flights_df) * 100):.2f}%",
            "avg_delay": f"{self.flights_df['ARRIVAL_DELAY'].mean():.2f} min"
        }

