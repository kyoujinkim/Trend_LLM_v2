"""
Cluster Analyzer
Analyzes cluster statistics, relationships, and characteristics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
import json
import os

from tqdm import tqdm


class ClusterAnalyzer:
    """Analyze cluster statistics and relationships"""

    def __init__(self):
        self.cluster_stats = {}
        self.cluster_relationships = {}

    def calculate_cluster_statistics(
        self,
        documents: pd.DataFrame,
        embeddings: np.ndarray
    ) -> Dict:
        """
        Calculate various statistics for each cluster

        Args:
            documents: DataFrame with documents and topic assignments
            embeddings: Document embeddings

        Returns:
            Dictionary with cluster statistics
        """
        stats = {}

        unique_topics = sorted([t for t in documents['Topic'].unique() if t != -1])

        for topic in tqdm(unique_topics):
            topic_docs = documents[documents['Topic'] == topic]
            topic_indices = topic_docs.index.tolist()
            topic_embeddings = embeddings[topic_indices]

            # Basic statistics
            cluster_size = len(topic_docs)

            # Embedding-based statistics
            centroid = topic_embeddings.mean(axis=0)
            distances = np.linalg.norm(topic_embeddings - centroid, axis=1)

            # Cohesion: average distance to centroid (lower is more cohesive)
            cohesion = distances.mean()
            cohesion_std = distances.std()

            # Density: inverse of average pairwise distance
            if cluster_size > 1:
                pairwise_sim = cosine_similarity(topic_embeddings)
                # Exclude diagonal
                mask = ~np.eye(pairwise_sim.shape[0], dtype=bool)
                avg_similarity = pairwise_sim[mask].mean()
            else:
                avg_similarity = 1.0

            # Subtopic diversity
            subtopic_counts = topic_docs['Subtopic'].value_counts()
            n_subtopics = len([st for st in subtopic_counts.index if st != -1])

            # Calculate entropy of subtopic distribution (higher = more diverse)
            if n_subtopics > 0:
                subtopic_probs = subtopic_counts.values / subtopic_counts.values.sum()
                subtopic_entropy = entropy(subtopic_probs)
            else:
                subtopic_entropy = 0.0

            subtopic_stat = {}
            unique_subtopics = sorted([t for t in topic_docs['Subtopic'].unique() if t != -1])
            for subtopic in unique_subtopics:
                subtopic_docs = topic_docs[topic_docs['Subtopic'] == subtopic]
                subtopic_indices = subtopic_docs.index.tolist()
                subtopic_embeddings = embeddings[subtopic_indices]

                # Additional subtopic-level statistics can be calculated here if needed
                pairwise_sim = cosine_similarity(subtopic_embeddings)
                # Exclude diagonal
                mask = ~np.eye(pairwise_sim.shape[0], dtype=bool)
                subtopic_avg_similarity = pairwise_sim[mask].mean()
                subtopic_stat[subtopic] = {
                    'size': len(subtopic_docs),
                    'avg_similarity': float(subtopic_avg_similarity)
                }

            # Time-based statistics if date information is available
            if 'date' in topic_docs.columns:
                topic_docs_copy = topic_docs.copy()
                topic_docs_copy['date_dt'] = pd.to_datetime(topic_docs_copy['date'], format='%Y%m%d', errors='coerce')
                date_range = (
                    topic_docs_copy['date_dt'].max() - topic_docs_copy['date_dt'].min()
                ).days if topic_docs_copy['date_dt'].notna().any() else 0

                # Activity over time
                daily_counts = topic_docs_copy.groupby('date_dt').size()
                activity_mean = daily_counts.mean() if len(daily_counts) > 0 else 0
                activity_std = daily_counts.std() if len(daily_counts) > 0 else 0
            else:
                date_range = None
                activity_mean = None
                activity_std = None

            stats[topic] = {
                'size': cluster_size,
                'cohesion': float(cohesion),
                'cohesion_std': float(cohesion_std),
                'avg_similarity': float(avg_similarity),
                'n_subtopics': n_subtopics,
                'subtopic_entropy': float(subtopic_entropy),
                'date_range_days': date_range,
                'daily_activity_mean': float(activity_mean) if activity_mean is not None else None,
                'daily_activity_std': float(activity_std) if activity_std is not None else None,
                'subtopic_stats': subtopic_stat
            }

        self.cluster_stats = stats
        return stats

    def find_cluster_relationships(
        self,
        documents: pd.DataFrame,
        embeddings: np.ndarray,
        top_k: int = 5
    ) -> Dict:
        """
        Find relationships between clusters based on embedding similarity

        Args:
            documents: DataFrame with documents and topic assignments
            embeddings: Document embeddings
            top_k: Number of top similar clusters to return for each cluster

        Returns:
            Dictionary mapping each cluster to its most similar clusters
        """
        relationships = {}

        unique_topics = sorted([t for t in documents['Topic'].unique() if t != -1])

        # Calculate cluster centroids
        centroids = {}
        for topic in unique_topics:
            topic_indices = documents[documents['Topic'] == topic].index.tolist()
            topic_embeddings = embeddings[topic_indices]
            centroids[topic] = topic_embeddings.mean(axis=0)

        # Calculate pairwise similarities
        centroid_list = [centroids[t] for t in unique_topics]
        centroid_matrix = np.array(centroid_list)
        similarity_matrix = cosine_similarity(centroid_matrix)

        # Find top-k similar clusters for each cluster
        for i, topic in enumerate(unique_topics):
            similarities = similarity_matrix[i]
            # Exclude self-similarity
            similarities[i] = -1

            # Get top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            relationships[topic] = [
                {
                    'topic': unique_topics[idx],
                    'similarity': float(similarities[idx])
                }
                for idx in top_indices
                if similarities[idx] > 0
            ]

        self.cluster_relationships = relationships
        return relationships

    def filter_clusters_by_stat(self, max_topic_size, max_subtopic_size):
        """
        Filter clusters and subtopics based on avg_similarity and size>10.
        Args:
            max_topic_size: maximum size of topic to keep
            max_subtopic_size: maximum size of subtopic to keep
        """
        if not self.cluster_stats:
            return

        filtered_stats = {}
        for topic, stats in tqdm(self.cluster_stats.items()):
            if stats['avg_similarity'] > 0.5 and stats['size'] > 10:
                # Filter subtopics
                sub_topic = pd.DataFrame(stats['subtopic_stats']).T
                if len(sub_topic)>0:
                    sub_topic = sub_topic.sort_values('size')
                    sub_topic = sub_topic[sub_topic['size'] > 10].iloc[:max_subtopic_size]
                    filtered_subtopics = sub_topic.to_dict(orient='index')

                stats['subtopic_stats'] = filtered_subtopics
                filtered_stats[topic] = stats

        filtered_stats_df = pd.DataFrame(filtered_stats).T.iloc[:max_topic_size]
        filtered_stats = filtered_stats_df.to_dict(orient='index')

        self.cluster_stats = filtered_stats

        return filtered_stats


    def analyze_cluster_overlap(
        self,
        documents: pd.DataFrame,
        field: str = 'source'
    ) -> Dict:
        """
        Analyze how clusters overlap in terms of a specific field (e.g., source, date)

        Args:
            documents: DataFrame with documents and topic assignments
            field: Column name to analyze overlap

        Returns:
            Dictionary with overlap analysis
        """
        if field not in documents.columns:
            return {}

        overlap_stats = {}
        unique_topics = sorted([t for t in documents['Topic'].unique() if t != -1])

        for topic in unique_topics:
            topic_docs = documents[documents['Topic'] == topic]
            field_distribution = topic_docs[field].value_counts().to_dict()

            overlap_stats[topic] = {
                'field': field,
                'distribution': field_distribution,
                'n_unique': len(field_distribution),
                'dominant': max(field_distribution, key=field_distribution.get) if field_distribution else None,
                'dominant_ratio': max(field_distribution.values()) / len(topic_docs) if field_distribution else 0
            }

        return overlap_stats

    def save_analysis(self, output_path: str):
        """
        Save all analysis results to files

        Args:
            output_path: Directory to save analysis results
        """
        os.makedirs(output_path, exist_ok=True)

        # Save cluster statistics
        if self.cluster_stats:
            with open(os.path.join(output_path, 'cluster_statistics.json'), 'w', encoding='utf-8') as f:
                json.dump(self.cluster_stats, f, ensure_ascii=False, indent=2)
            print(f"Cluster statistics saved to {output_path}/cluster_statistics.json")

        # Save cluster relationships
        if self.cluster_relationships:
            with open(os.path.join(output_path, 'cluster_relationships.json'), 'w', encoding='utf-8') as f:
                json.dump(self.cluster_relationships, f, ensure_ascii=False, indent=2)
            print(f"Cluster relationships saved to {output_path}/cluster_relationships.json")

    def generate_summary_report(self) -> str:
        """
        Generate a text summary of cluster analysis

        Returns:
            String with summary report
        """
        report = []
        report.append("="*80)
        report.append("CLUSTER ANALYSIS SUMMARY")
        report.append("="*80)
        report.append("")

        if self.cluster_stats:
            report.append(f"Total clusters analyzed: {len(self.cluster_stats)}")
            report.append("")

            # Overall statistics
            all_sizes = [s['size'] for s in self.cluster_stats.values()]
            all_cohesions = [s['cohesion'] for s in self.cluster_stats.values()]
            all_similarities = [s['avg_similarity'] for s in self.cluster_stats.values()]

            report.append("Overall Statistics:")
            report.append(f"  Average cluster size: {np.mean(all_sizes):.1f} (std: {np.std(all_sizes):.1f})")
            report.append(f"  Average cohesion: {np.mean(all_cohesions):.3f} (std: {np.std(all_cohesions):.3f})")
            report.append(f"  Average similarity: {np.mean(all_similarities):.3f} (std: {np.std(all_similarities):.3f})")
            report.append("")

            # Top 5 largest clusters
            sorted_by_size = sorted(self.cluster_stats.items(), key=lambda x: x[1]['size'], reverse=True)
            report.append("Top 5 Largest Clusters:")
            for topic, stats in sorted_by_size[:5]:
                report.append(f"  Topic {topic}: {stats['size']} documents, "
                            f"{stats['n_subtopics']} subtopics, "
                            f"similarity: {stats['avg_similarity']:.3f}")
            report.append("")

            # Most cohesive clusters
            sorted_by_similarity = sorted(self.cluster_stats.items(),
                                         key=lambda x: x[1]['avg_similarity'], reverse=True)
            report.append("Top 5 Most Cohesive Clusters:")
            for topic, stats in sorted_by_similarity[:5]:
                report.append(f"  Topic {topic}: similarity {stats['avg_similarity']:.3f}, "
                            f"size: {stats['size']}")
            report.append("")

        report.append("="*80)
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    import pickle
    from glob import glob

    # Load sample data
    doc_list = glob('./data/news/*.csv')
    if doc_list:
        documents = pd.concat([pd.read_csv(f, encoding='UTF-8-sig') for f in doc_list], axis=0)

        # Assume embeddings and topics are already assigned
        # documents['Topic'] = ...
        # embeddings = ...

        analyzer = ClusterAnalyzer()
        # stats = analyzer.calculate_cluster_statistics(documents, embeddings)
        # relationships = analyzer.find_cluster_relationships(documents, embeddings)
        # analyzer.save_analysis('./data/output/analysis')
        # print(analyzer.generate_summary_report())