"""
Cluster Analyzer Module
Analyzes cluster statistics, relationships, and evolution over time
"""

import json
import pickle
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from collections import defaultdict


class ClusterAnalyzer:
    """Analyze cluster statistics and relationships"""

    def __init__(self):
        self.cluster_stats = {}
        self.cluster_relationships = {}
        self.cluster_evolution = []

    def calculate_cluster_statistics(self, topic_model, documents_df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive statistics for each cluster.

        Args:
            topic_model: Fitted topic model with topic representations
            documents_df: DataFrame with documents and topic assignments

        Returns:
            Dictionary containing statistics for each topic
        """
        stats = {}

        for topic_id in topic_model.topic_representations_.keys():
            topic_docs = documents_df[documents_df['Topic'] == topic_id]

            # Basic statistics
            stats[topic_id] = {
                'size': len(topic_docs),
                'percentage': len(topic_docs) / len(documents_df) * 100,
                'top_words': [w[0] for w in topic_model.topic_representations_[topic_id][:5]],
                'label': topic_model.topic_labels_[topic_id]
            }

            # If probabilities available, calculate confidence metrics
            if topic_model.probabilities_ is not None:
                topic_probs = topic_model.probabilities_[topic_docs.index]
                stats[topic_id].update({
                    'avg_probability': float(np.mean(topic_probs)),
                    'std_probability': float(np.std(topic_probs)),
                    'min_probability': float(np.min(topic_probs)),
                    'max_probability': float(np.max(topic_probs))
                })

        self.cluster_stats = stats
        return stats

    def find_cluster_relationships(self, topic_model, similarity_threshold: float = 0.3) -> Dict:
        """
        Find relationships between clusters based on topic representation similarity.

        Args:
            topic_model: Fitted topic model
            similarity_threshold: Minimum cosine similarity to consider clusters related

        Returns:
            Dictionary mapping topic pairs to their similarity scores
        """
        relationships = defaultdict(list)

        topics = list(topic_model.topic_representations_.keys())

        # Calculate pairwise similarities using c-TF-IDF vectors
        for i, topic_i in enumerate(topics):
            for topic_j in topics[i + 1:]:
                # Get c-TF-IDF vectors
                vec_i = topic_model.c_tf_idf_[i].toarray().flatten()
                vec_j = topic_model.c_tf_idf_[topics.index(topic_j)].toarray().flatten()

                # Calculate cosine similarity
                similarity = 1 - cosine(vec_i, vec_j)

                if similarity > similarity_threshold:
                    relationships[topic_i].append({
                        'related_topic': topic_j,
                        'similarity': float(similarity)
                    })
                    relationships[topic_j].append({
                        'related_topic': topic_i,
                        'similarity': float(similarity)
                    })

        self.cluster_relationships = dict(relationships)
        return self.cluster_relationships

    def track_cluster_changes(self,
                              prev_topics: Dict,
                              curr_topics: Dict,
                              prev_assignments: List[int],
                              curr_assignments: List[int],
                              timestamp: str) -> Dict:
        """
        Track changes in clusters over time (merge, split, grow, shrink).

        Args:
            prev_topics: Previous topic representations
            curr_topics: Current topic representations
            prev_assignments: Previous topic assignments
            curr_assignments: Current topic assignments
            timestamp: Timestamp for this comparison

        Returns:
            Dictionary describing cluster changes
        """
        changes = {
            'timestamp': timestamp,
            'new_topics': [],
            'disappeared_topics': [],
            'grown_topics': [],
            'shrunk_topics': [],
            'merged_topics': [],
            'split_topics': []
        }

        # Count topic sizes
        prev_sizes = pd.Series(prev_assignments).value_counts().to_dict()
        curr_sizes = pd.Series(curr_assignments).value_counts().to_dict()

        # Find new and disappeared topics
        prev_topic_ids = set(prev_topics.keys())
        curr_topic_ids = set(curr_topics.keys())

        changes['new_topics'] = list(curr_topic_ids - prev_topic_ids)
        changes['disappeared_topics'] = list(prev_topic_ids - curr_topic_ids)

        # Track size changes for existing topics
        for topic_id in prev_topic_ids & curr_topic_ids:
            prev_size = prev_sizes.get(topic_id, 0)
            curr_size = curr_sizes.get(topic_id, 0)

            size_change = ((curr_size - prev_size) / prev_size * 100) if prev_size > 0 else 0

            if size_change > 20:  # 20% growth threshold
                changes['grown_topics'].append({
                    'topic_id': topic_id,
                    'prev_size': prev_size,
                    'curr_size': curr_size,
                    'growth_pct': size_change
                })
            elif size_change < -20:  # 20% shrink threshold
                changes['shrunk_topics'].append({
                    'topic_id': topic_id,
                    'prev_size': prev_size,
                    'curr_size': curr_size,
                    'shrink_pct': size_change
                })

        self.cluster_evolution.append(changes)
        return changes

    def calculate_topic_diversity(self, topic_model) -> Dict:
        """
        Calculate diversity metrics for each topic.

        Args:
            topic_model: Fitted topic model

        Returns:
            Dictionary with diversity metrics per topic
        """
        diversity = {}

        for topic_id, words in topic_model.topic_representations_.items():
            # Get word probabilities
            word_scores = np.array([w[1] for w in words])

            # Normalize to create probability distribution
            word_probs = word_scores / word_scores.sum()

            # Calculate entropy as diversity measure
            topic_entropy = entropy(word_probs)

            diversity[topic_id] = {
                'entropy': float(topic_entropy),
                'unique_words': len(words),
                'concentration': float(word_probs[0])  # Weight of top word
            }

        return diversity

    def save_analysis(self, output_path: str):
        """
        Save analysis results to files.

        Args:
            output_path: Directory to save analysis results
        """
        import os
        os.makedirs(output_path, exist_ok=True)

        # Save cluster statistics
        with open(f'{output_path}/cluster_statistics.json', 'w', encoding='utf-8') as f:
            json.dump(self.cluster_stats, f, ensure_ascii=False, indent=2)

        # Save cluster relationships
        with open(f'{output_path}/cluster_relationships.json', 'w', encoding='utf-8') as f:
            json.dump(self.cluster_relationships, f, ensure_ascii=False, indent=2)

        # Save cluster evolution
        if self.cluster_evolution:
            with open(f'{output_path}/cluster_evolution.json', 'w', encoding='utf-8') as f:
                json.dump(self.cluster_evolution, f, ensure_ascii=False, indent=2)

        print(f"Analysis results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Cluster Analyzer module - use as import")