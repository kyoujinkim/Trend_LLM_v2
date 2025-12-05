"""
Cluster Tracker
Tracks cluster evolution over time, including merge/split detection and stability analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from scipy.fft import fft, fftfreq
import json
import os
from collections import defaultdict

from tqdm import tqdm


class ClusterTracker:
    """Track cluster evolution and changes over time"""

    def __init__(self, time_window: int = 7):
        """
        Initialize cluster tracker

        Args:
            time_window: Number of days to use for rolling window analysis
        """
        self.time_window = time_window
        self.cluster_timeline = {}
        self.evolution_events = []

    def create_cluster_timeline(self, documents: pd.DataFrame, embeddings: np.ndarray) -> Dict:
        """
        Create timeline of cluster evolution

        Args:
            documents: DataFrame with documents, topics, and dates
            embeddings: Document embeddings

        Returns:
            Dictionary with cluster timeline information
        """
        if 'date' not in documents.columns:
            print("Warning: No date column found. Cannot create timeline.")
            return {}

        # Convert date to datetime
        documents = documents.copy()
        documents['date_dt'] = pd.to_datetime(documents['date'], format='%Y%m%d', errors='coerce')
        documents = documents[documents['date_dt'].notna()]

        # Get unique dates and topics
        unique_dates = sorted(documents['date_dt'].unique())
        unique_topics = sorted([t for t in documents['Topic'].unique() if t != -1])

        timeline = {}

        # For each topic, track its evolution over time
        for topic in unique_topics:
            topic_docs = documents[documents['Topic'] == topic]
            topic_indices = topic_docs.index.tolist()
            topic_embeddings = embeddings[topic_indices]

            # Calculate centroid
            centroid = topic_embeddings.mean(axis=0)

            # Track metrics over time
            daily_metrics = []

            for date in unique_dates:
                date_topic_docs = topic_docs[topic_docs['date_dt'] == date]

                if len(date_topic_docs) > 0:
                    date_topic_indices = date_topic_docs.index.tolist()
                    date_embeddings = embeddings[date_topic_indices]

                    # Calculate daily centroid
                    daily_centroid = date_embeddings.mean(axis=0)

                    # Calculate drift from overall centroid
                    drift = float(1 - cosine_similarity([centroid], [daily_centroid])[0, 0])

                    # Calculate daily cohesion
                    distances = np.linalg.norm(date_embeddings - daily_centroid, axis=1)
                    cohesion = float(distances.mean())

                    daily_metrics.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'count': len(date_topic_docs),
                        'drift': drift,
                        'cohesion': cohesion
                    })
                else:
                    daily_metrics.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'count': 0,
                        'drift': None,
                        'cohesion': None
                    })

            timeline[topic] = {
                'daily_metrics': daily_metrics,
                'total_documents': len(topic_docs),
                'date_range': {
                    'start': topic_docs['date_dt'].min().strftime('%Y-%m-%d'),
                    'end': topic_docs['date_dt'].max().strftime('%Y-%m-%d'),
                    'days': (topic_docs['date_dt'].max() - topic_docs['date_dt'].min()).days
                }
            }

        self.cluster_timeline = timeline
        return timeline

    def detect_evolution_events(
        self,
        documents: pd.DataFrame,
        embeddings: np.ndarray,
        growth_threshold: float = 2.0,
        shrink_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Detect cluster evolution events (growth, shrinkage, emergence, disappearance)

        Args:
            documents: DataFrame with documents, topics, and dates
            embeddings: Document embeddings
            growth_threshold: Ratio threshold for detecting growth
            shrink_threshold: Ratio threshold for detecting shrinkage

        Returns:
            List of detected events
        """
        if not self.cluster_timeline:
            self.create_cluster_timeline(documents, embeddings)

        events = []

        for topic, timeline_data in self.cluster_timeline.items():
            daily_metrics = timeline_data['daily_metrics']
            counts = [m['count'] for m in daily_metrics]

            # Skip if not enough data
            if len(counts) < self.time_window:
                continue

            # Calculate rolling averages
            for i in range(self.time_window, len(counts)):
                prev_avg = np.mean(counts[i-self.time_window:i])
                curr_count = counts[i]

                if prev_avg > 0:
                    ratio = curr_count / prev_avg

                    # Detect growth
                    if ratio >= growth_threshold:
                        events.append({
                            'type': 'growth',
                            'topic': topic,
                            'date': daily_metrics[i]['date'],
                            'ratio': float(ratio),
                            'prev_avg': float(prev_avg),
                            'current': curr_count
                        })

                    # Detect shrinkage
                    elif ratio <= shrink_threshold and ratio > 0:
                        events.append({
                            'type': 'shrinkage',
                            'topic': topic,
                            'date': daily_metrics[i]['date'],
                            'ratio': float(ratio),
                            'prev_avg': float(prev_avg),
                            'current': curr_count
                        })

            # Detect emergence (first appearance)
            first_nonzero = next((i for i, c in enumerate(counts) if c > 0), None)
            if first_nonzero is not None and first_nonzero > 0:
                events.append({
                    'type': 'emergence',
                    'topic': topic,
                    'date': daily_metrics[first_nonzero]['date'],
                    'count': counts[first_nonzero]
                })

            # Detect disappearance (last appearance)
            last_nonzero = len(counts) - 1 - next((i for i, c in enumerate(reversed(counts)) if c > 0), len(counts) - 1)
            if last_nonzero < len(counts) - self.time_window:
                events.append({
                    'type': 'disappearance',
                    'topic': topic,
                    'date': daily_metrics[last_nonzero]['date'],
                    'last_count': counts[last_nonzero]
                })

        self.evolution_events = events
        return events

    def analyze_cluster_stability(
        self,
        use_fft: bool = True
    ) -> Dict:
        """
        Analyze cluster stability using temporal patterns

        Args:
            use_fft: Whether to use FFT for frequency analysis

        Returns:
            Dictionary with stability metrics for each cluster
        """
        if not self.cluster_timeline:
            print("Warning: Timeline not created. Cannot analyze stability.")
            return {}

        stability_metrics = {}

        for topic, timeline_data in self.cluster_timeline.items():
            daily_metrics = timeline_data['daily_metrics']
            counts = np.array([m['count'] for m in daily_metrics])
            drifts = np.array([m['drift'] if m['drift'] is not None else 0 for m in daily_metrics])

            # Basic stability metrics
            count_std = float(np.std(counts))
            count_mean = float(np.mean(counts))
            cv = count_std / count_mean if count_mean > 0 else 0  # Coefficient of variation

            drift_mean = float(np.mean(drifts[drifts > 0])) if len(drifts[drifts > 0]) > 0 else 0
            drift_std = float(np.std(drifts[drifts > 0])) if len(drifts[drifts > 0]) > 0 else 0

            stability_metrics[topic] = {
                'count_mean': count_mean,
                'count_std': count_std,
                'coefficient_of_variation': float(cv),
                'drift_mean': drift_mean,
                'drift_std': drift_std,
                'stability_score': float(1.0 / (1.0 + cv + drift_mean))  # Higher = more stable
            }

            # FFT analysis for periodicity
            if use_fft and len(counts) > 10:
                # Apply FFT to count time series
                fft_vals = fft(counts)
                freqs = fftfreq(len(counts), d=1)  # Daily frequency

                # Get magnitude spectrum
                magnitude = np.abs(fft_vals)

                # Find dominant frequencies (excluding DC component)
                positive_freqs = freqs[1:len(freqs)//2]
                positive_magnitude = magnitude[1:len(freqs)//2]

                if len(positive_magnitude) > 0:
                    dominant_freq_idx = np.argmax(positive_magnitude)
                    dominant_freq = float(positive_freqs[dominant_freq_idx])
                    dominant_period = 1.0 / dominant_freq if dominant_freq != 0 else 0

                    # Power spectral density
                    total_power = float(np.sum(positive_magnitude ** 2))
                    dominant_power = float(positive_magnitude[dominant_freq_idx] ** 2)
                    power_ratio = dominant_power / total_power if total_power > 0 else 0

                    stability_metrics[topic]['fft_analysis'] = {
                        'dominant_frequency': dominant_freq,
                        'dominant_period_days': float(dominant_period),
                        'power_ratio': float(power_ratio),
                        'is_periodic': power_ratio > 0.3  # Threshold for periodicity
                    }

        return stability_metrics

    def detect_merge_split_events(
        self,
        documents: pd.DataFrame,
        embeddings: np.ndarray,
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Detect potential merge and split events between clusters over time

        Args:
            documents: DataFrame with documents, topics, and dates
            embeddings: Document embeddings
            similarity_threshold: Cosine similarity threshold for merge/split detection

        Returns:
            List of detected merge/split events
        """
        if 'date' not in documents.columns:
            return []

        documents = documents.copy()
        documents['date_dt'] = pd.to_datetime(documents['date'], format='%Y%m%d', errors='coerce')
        documents = documents[documents['date_dt'].notna()]

        unique_dates = sorted(documents['date_dt'].unique())
        merge_split_events = []

        # Track cluster centroids over time windows
        window_size = self.time_window

        for i in tqdm(range(len(unique_dates) - window_size)):
            # Define two time windows
            window1_dates = unique_dates[i:i+window_size]
            window2_dates = unique_dates[i+window_size:i+2*window_size]

            if len(window2_dates) < window_size:
                break

            # Get documents in each window
            window1_docs = documents[documents['date_dt'].isin(window1_dates)]
            window2_docs = documents[documents['date_dt'].isin(window2_dates)]

            # Calculate centroids for topics in each window
            centroids1 = self._calculate_topic_centroids(window1_docs, embeddings)
            centroids2 = self._calculate_topic_centroids(window2_docs, embeddings)

            # Compare topic configurations
            topics1 = set(centroids1.keys())
            topics2 = set(centroids2.keys())

            # New topics in window 2
            new_topics = topics2 - topics1

            # Check if new topics are similar to combinations of old topics (potential merge)
            for new_topic in new_topics:
                new_centroid = centroids2[new_topic]

                # Check similarity with topics from window 1
                similar_topics = []
                for old_topic in topics1:
                    old_centroid = centroids1[old_topic]
                    sim = cosine_similarity([new_centroid], [old_centroid])[0, 0]
                    if sim > similarity_threshold:
                        similar_topics.append((old_topic, float(sim)))

                if len(similar_topics) >= 2:
                    merge_split_events.append({
                        'type': 'potential_merge',
                        'date_range': f"{window2_dates[0].strftime('%Y-%m-%d')} to {window2_dates[-1].strftime('%Y-%m-%d')}",
                        'new_topic': new_topic,
                        'source_topics': similar_topics
                    })

            # Disappeared topics (potential split)
            disappeared_topics = topics1 - topics2

            for old_topic in disappeared_topics:
                old_centroid = centroids1[old_topic]

                # Check if it split into multiple new topics
                similar_new_topics = []
                for new_topic in new_topics:
                    new_centroid = centroids2[new_topic]
                    sim = cosine_similarity([old_centroid], [new_centroid])[0, 0]
                    if sim > similarity_threshold:
                        similar_new_topics.append((new_topic, float(sim)))

                if len(similar_new_topics) >= 2:
                    merge_split_events.append({
                        'type': 'potential_split',
                        'date_range': f"{window2_dates[0].strftime('%Y-%m-%d')} to {window2_dates[-1].strftime('%Y-%m-%d')}",
                        'old_topic': old_topic,
                        'new_topics': similar_new_topics
                    })

        return merge_split_events

    def _calculate_topic_centroids(
        self,
        documents: pd.DataFrame,
        embeddings: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """Helper method to calculate centroids for all topics in a document set"""
        centroids = {}
        unique_topics = [t for t in documents['Topic'].unique() if t != -1]

        for topic in unique_topics:
            topic_docs = documents[documents['Topic'] == topic]
            topic_indices = [documents.index.get_loc(idx) for idx in topic_docs.index]
            topic_embeddings = embeddings[topic_indices]
            centroids[topic] = topic_embeddings.mean(axis=0)

        return centroids

if __name__ == "__main__":
    # Example usage
    import pickle
    from glob import glob

    # Load sample data
    doc_list = glob('./data/news/*.csv')
    if doc_list:
        documents = pd.concat([pd.read_csv(f, encoding='UTF-8-sig') for f in doc_list], axis=0)

        # Assume embeddings and topics are already assigned
        # tracker = ClusterTracker(time_window=7)
        # timeline = tracker.create_cluster_timeline(documents, embeddings)
        # events = tracker.detect_evolution_events(documents, embeddings)
        # stability = tracker.analyze_cluster_stability(use_fft=True)
        # tracker.save_tracking_results('./data/output/tracking')