"""
Time Series Tracker Module
Tracks cluster evolution over time and analyzes temporal patterns
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import fft
from collections import defaultdict
import pickle


class TimeSeriesTracker:
    """Track and analyze cluster evolution over time"""

    def __init__(self):
        self.time_series_data = defaultdict(lambda: defaultdict(list))
        self.cluster_lifespans = {}
        self.stability_metrics = {}

    def track_cluster_over_time(self,
                                 documents_df: pd.DataFrame,
                                 date_column: str = 'date_dt',
                                 freq: str = 'D') -> Dict:
        """
        Track cluster frequencies over time.

        Args:
            documents_df: DataFrame with documents, topics, and dates
            date_column: Name of the date column
            freq: Frequency for aggregation ('D' for daily, 'W' for weekly, 'M' for monthly)

        Returns:
            Dictionary with time series data for each topic
        """
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(documents_df[date_column]):
            documents_df[date_column] = pd.to_datetime(documents_df[date_column])

        # Group by date and topic
        time_series = documents_df.groupby([
            pd.Grouper(key=date_column, freq=freq),
            'Topic'
        ]).size().reset_index(name='count')

        # Pivot to get topics as columns
        pivot_data = time_series.pivot(index=time_series.columns[0], columns='Topic', values='count')
        pivot_data = pivot_data.fillna(0)

        # Store time series for each topic
        for topic_id in pivot_data.columns:
            if topic_id != -1:  # Exclude noise cluster
                self.time_series_data[topic_id] = {
                    'dates': pivot_data.index.astype(str).tolist(),
                    'counts': pivot_data[topic_id].tolist(),
                    'total_documents': int(pivot_data[topic_id].sum()),
                    'avg_daily_count': float(pivot_data[topic_id].mean()),
                    'max_count': int(pivot_data[topic_id].max()),
                    'min_count': int(pivot_data[topic_id].min())
                }

        return dict(self.time_series_data)

    def calculate_cluster_lifespan(self,
                                    documents_df: pd.DataFrame,
                                    date_column: str = 'date_dt') -> Dict:
        """
        Calculate the lifespan of each cluster.

        Args:
            documents_df: DataFrame with documents, topics, and dates
            date_column: Name of the date column

        Returns:
            Dictionary with lifespan information for each topic
        """
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(documents_df[date_column]):
            documents_df[date_column] = pd.to_datetime(documents_df[date_column])

        for topic_id in documents_df['Topic'].unique():
            if topic_id != -1:  # Exclude noise cluster
                topic_dates = documents_df[documents_df['Topic'] == topic_id][date_column]

                if len(topic_dates) > 0:
                    first_appearance = topic_dates.min()
                    last_appearance = topic_dates.max()
                    lifespan_days = (last_appearance - first_appearance).days

                    self.cluster_lifespans[topic_id] = {
                        'first_appearance': first_appearance.strftime('%Y-%m-%d'),
                        'last_appearance': last_appearance.strftime('%Y-%m-%d'),
                        'lifespan_days': lifespan_days,
                        'document_count': len(topic_dates),
                        'active_days': len(topic_dates.dt.date.unique())
                    }

        return self.cluster_lifespans

    def analyze_cluster_stability(self, topic_id: int, method: str = 'variance') -> Dict:
        """
        Analyze cluster stability over time.

        Args:
            topic_id: Topic ID to analyze
            method: Method for stability analysis ('variance', 'fft', 'cv')

        Returns:
            Dictionary with stability metrics
        """
        if topic_id not in self.time_series_data:
            return {}

        counts = np.array(self.time_series_data[topic_id]['counts'])

        stability = {
            'topic_id': topic_id,
            'method': method
        }

        if method == 'variance':
            # Lower variance indicates more stability
            stability['variance'] = float(np.var(counts))
            stability['std_dev'] = float(np.std(counts))
            stability['coefficient_of_variation'] = float(np.std(counts) / np.mean(counts)) if np.mean(counts) > 0 else 0

        elif method == 'fft':
            # FFT to detect periodicity and trends
            if len(counts) > 1:
                fft_result = fft.fft(counts)
                frequencies = fft.fftfreq(len(counts))

                # Get dominant frequencies
                magnitude = np.abs(fft_result)
                dominant_idx = np.argsort(magnitude)[-3:]  # Top 3 frequencies

                stability['dominant_frequencies'] = frequencies[dominant_idx].tolist()
                stability['frequency_magnitudes'] = magnitude[dominant_idx].tolist()
                stability['has_strong_periodicity'] = bool(np.max(magnitude[1:]) > 2 * np.mean(magnitude[1:]))

        elif method == 'cv':
            # Coefficient of variation
            if np.mean(counts) > 0:
                stability['coefficient_of_variation'] = float(np.std(counts) / np.mean(counts))
            else:
                stability['coefficient_of_variation'] = 0

        return stability

    def analyze_all_stabilities(self, method: str = 'variance') -> Dict:
        """
        Analyze stability for all tracked topics.

        Args:
            method: Method for stability analysis

        Returns:
            Dictionary with stability metrics for all topics
        """
        for topic_id in self.time_series_data.keys():
            self.stability_metrics[topic_id] = self.analyze_cluster_stability(topic_id, method)

        return self.stability_metrics

    def detect_trending_topics(self,
                                window_size: int = 7,
                                growth_threshold: float = 50.0) -> List[Dict]:
        """
        Detect topics that are trending (rapidly growing).

        Args:
            window_size: Number of time periods to compare
            growth_threshold: Minimum percentage growth to consider trending

        Returns:
            List of trending topics with their growth metrics
        """
        trending = []

        for topic_id, data in self.time_series_data.items():
            counts = np.array(data['counts'])

            if len(counts) >= window_size * 2:
                # Compare recent window with previous window
                recent_avg = np.mean(counts[-window_size:])
                previous_avg = np.mean(counts[-window_size*2:-window_size])

                if previous_avg > 0:
                    growth_pct = ((recent_avg - previous_avg) / previous_avg) * 100

                    if growth_pct >= growth_threshold:
                        trending.append({
                            'topic_id': topic_id,
                            'growth_percentage': float(growth_pct),
                            'recent_avg': float(recent_avg),
                            'previous_avg': float(previous_avg)
                        })

        # Sort by growth percentage
        trending.sort(key=lambda x: x['growth_percentage'], reverse=True)
        return trending

    def detect_declining_topics(self,
                                 window_size: int = 7,
                                 decline_threshold: float = -30.0) -> List[Dict]:
        """
        Detect topics that are declining.

        Args:
            window_size: Number of time periods to compare
            decline_threshold: Maximum percentage decline to consider declining

        Returns:
            List of declining topics with their decline metrics
        """
        declining = []

        for topic_id, data in self.time_series_data.items():
            counts = np.array(data['counts'])

            if len(counts) >= window_size * 2:
                # Compare recent window with previous window
                recent_avg = np.mean(counts[-window_size:])
                previous_avg = np.mean(counts[-window_size*2:-window_size])

                if previous_avg > 0:
                    change_pct = ((recent_avg - previous_avg) / previous_avg) * 100

                    if change_pct <= decline_threshold:
                        declining.append({
                            'topic_id': topic_id,
                            'decline_percentage': float(change_pct),
                            'recent_avg': float(recent_avg),
                            'previous_avg': float(previous_avg)
                        })

        # Sort by decline percentage
        declining.sort(key=lambda x: x['decline_percentage'])
        return declining

    def save_time_series_data(self, output_path: str):
        """
        Save time series data and analysis results.

        Args:
            output_path: Directory to save results
        """
        import os
        os.makedirs(output_path, exist_ok=True)

        # Save time series data
        with open(f'{output_path}/time_series_data.json', 'w', encoding='utf-8') as f:
            json.dump(dict(self.time_series_data), f, ensure_ascii=False, indent=2)

        # Save cluster lifespans
        if self.cluster_lifespans:
            with open(f'{output_path}/cluster_lifespans.json', 'w', encoding='utf-8') as f:
                json.dump(self.cluster_lifespans, f, ensure_ascii=False, indent=2)

        # Save stability metrics
        if self.stability_metrics:
            with open(f'{output_path}/stability_metrics.json', 'w', encoding='utf-8') as f:
                json.dump(self.stability_metrics, f, ensure_ascii=False, indent=2)

        # Detect and save trending/declining topics
        trending = self.detect_trending_topics()
        declining = self.detect_declining_topics()

        with open(f'{output_path}/trending_topics.json', 'w', encoding='utf-8') as f:
            json.dump(trending, f, ensure_ascii=False, indent=2)

        with open(f'{output_path}/declining_topics.json', 'w', encoding='utf-8') as f:
            json.dump(declining, f, ensure_ascii=False, indent=2)

        print(f"Time series data saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Time Series Tracker module - use as import")