"""
Custom Analysis Example
Demonstrates how to perform custom analyses on clustering results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


def load_results(output_path='./data/output'):
    """Load pipeline results"""
    output_path = Path(output_path)

    # Load documents and embeddings
    with open(output_path / 'cache.pkl', 'rb') as f:
        cache = pickle.load(f)

    # Load keywords
    with open(output_path / 'topic_keywords.json', 'r', encoding='utf-8') as f:
        topic_keywords = json.load(f)

    with open(output_path / 'subtopic_keywords.json', 'r', encoding='utf-8') as f:
        subtopic_keywords = json.load(f)

    return cache['documents'], cache['embeddings'], topic_keywords, subtopic_keywords


def example_1_find_related_topics(documents, embeddings, topic_id, top_k=5):
    """Find topics most related to a given topic"""
    print(f"\nExample 1: Finding topics related to Topic {topic_id}")
    print("-"*80)

    # Calculate topic centroids
    unique_topics = [t for t in documents['Topic'].unique() if t != -1]
    centroids = {}

    for tid in unique_topics:
        topic_indices = documents[documents['Topic'] == tid].index.tolist()
        topic_embeddings = embeddings[topic_indices]
        centroids[tid] = topic_embeddings.mean(axis=0)

    # Calculate similarities
    target_centroid = centroids[topic_id]
    similarities = {}

    for tid, centroid in centroids.items():
        if tid != topic_id:
            sim = cosine_similarity([target_centroid], [centroid])[0, 0]
            similarities[tid] = sim

    # Get top-k most similar
    top_related = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]

    print(f"\nTop {top_k} topics related to Topic {topic_id}:")
    for rank, (tid, sim) in enumerate(top_related, 1):
        size = len(documents[documents['Topic'] == tid])
        print(f"{rank}. Topic {tid} (similarity: {sim:.3f}, size: {size:,})")


def example_2_temporal_trends(documents):
    """Analyze temporal trends for topics"""
    print("\nExample 2: Temporal Trend Analysis")
    print("-"*80)

    if 'date' not in documents.columns:
        print("No date information available")
        return

    docs = documents.copy()
    docs['date_dt'] = pd.to_datetime(docs['date'], format='%Y%m%d', errors='coerce')
    docs = docs[docs['date_dt'].notna()]

    # Get top 5 topics
    top_topics = docs[docs['Topic'] != -1]['Topic'].value_counts().head(5).index

    print("\nWeekly growth rate for top 5 topics:")
    print(f"{'Topic':<10} {'Week 1':<10} {'Week 2':<10} {'Week 3':<10} {'Week 4':<10}")
    print("-"*60)

    for topic_id in top_topics:
        topic_docs = docs[docs['Topic'] == topic_id]
        topic_docs = topic_docs.set_index('date_dt').sort_index()

        # Resample by week
        weekly = topic_docs.resample('W').size()

        if len(weekly) >= 4:
            values = [f"{v:<10}" for v in weekly.head(4).values]
            print(f"{topic_id:<10} {' '.join(values)}")


def example_3_subtopic_diversity(documents, subtopic_keywords):
    """Analyze sub-topic diversity within main topics"""
    print("\nExample 3: Sub-topic Diversity Analysis")
    print("-"*80)

    topic_diversity = []

    for topic_id in documents[documents['Topic'] != -1]['Topic'].unique():
        topic_docs = documents[documents['Topic'] == topic_id]
        subtopics = topic_docs[topic_docs['Subtopic'] != -1]['Subtopic'].unique()

        n_subtopics = len(subtopics)
        topic_size = len(topic_docs)

        # Calculate entropy
        if n_subtopics > 0:
            subtopic_counts = topic_docs['Subtopic'].value_counts(normalize=True)
            entropy = -sum(p * np.log2(p) for p in subtopic_counts if p > 0)
        else:
            entropy = 0

        topic_diversity.append({
            'topic': topic_id,
            'size': topic_size,
            'n_subtopics': n_subtopics,
            'entropy': entropy
        })

    # Sort by entropy (most diverse first)
    topic_diversity.sort(key=lambda x: x['entropy'], reverse=True)

    print("\nMost diverse topics (high sub-topic entropy):")
    print(f"{'Topic':<10} {'Size':<10} {'Sub-topics':<12} {'Entropy':<10}")
    print("-"*60)

    for item in topic_diversity[:10]:
        print(f"{item['topic']:<10} {item['size']:<10} {item['n_subtopics']:<12} {item['entropy']:<10.2f}")


def example_4_keyword_cooccurrence(topic_keywords):
    """Analyze keyword co-occurrence across topics"""
    print("\nExample 4: Keyword Co-occurrence Analysis")
    print("-"*80)

    # Build keyword-topic matrix
    all_keywords = set()
    for keywords in topic_keywords.values():
        if isinstance(keywords, list):
            all_keywords.update(keywords)

    # Find most common keywords
    keyword_counts = {}
    for keywords in topic_keywords.values():
        if isinstance(keywords, list):
            for kw in keywords:
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1

    # Get top keywords
    top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    print("\nMost frequent keywords across all topics:")
    print(f"{'Keyword':<30} {'Frequency':<10}")
    print("-"*60)

    for kw, count in top_keywords:
        print(f"{kw:<30} {count:<10}")


def example_5_outlier_analysis(documents, embeddings):
    """Analyze outlier documents"""
    print("\nExample 5: Outlier Analysis")
    print("-"*80)

    outliers = documents[documents['Topic'] == -1]

    print(f"\nTotal outliers: {len(outliers):,}")
    print(f"Percentage of dataset: {100 * len(outliers) / len(documents):.1f}%")

    if len(outliers) == 0:
        return

    # Analyze if outliers cluster together
    outlier_indices = outliers.index.tolist()
    outlier_embeddings = embeddings[outlier_indices]

    # Calculate pairwise distances
    from sklearn.metrics.pairwise import euclidean_distances

    distances = euclidean_distances(outlier_embeddings)
    avg_distance = distances.mean()

    print(f"\nAverage distance between outliers: {avg_distance:.3f}")

    # Find nearest non-outlier topic for each outlier
    non_outliers = documents[documents['Topic'] != -1]
    topics = non_outliers['Topic'].unique()

    # Calculate topic centroids
    centroids = {}
    for tid in topics:
        topic_indices = documents[documents['Topic'] == tid].index.tolist()
        topic_embeddings = embeddings[topic_indices]
        centroids[tid] = topic_embeddings.mean(axis=0)

    # Find nearest topic for outliers
    nearest_topics = {}
    for idx, outlier_emb in zip(outlier_indices, outlier_embeddings):
        min_dist = float('inf')
        nearest_topic = None

        for tid, centroid in centroids.items():
            dist = np.linalg.norm(outlier_emb - centroid)
            if dist < min_dist:
                min_dist = dist
                nearest_topic = tid

        nearest_topics[nearest_topic] = nearest_topics.get(nearest_topic, 0) + 1

    print("\nOutliers' nearest topics (potential re-assignment candidates):")
    for tid, count in sorted(nearest_topics.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  Topic {tid}: {count} outliers")


def main():
    """Run all custom analysis examples"""
    print("="*80)
    print("CUSTOM ANALYSIS EXAMPLES")
    print("="*80)

    # Load results
    print("\nLoading results...")
    documents, embeddings, topic_keywords, subtopic_keywords = load_results()
    print(f"✓ Loaded {len(documents):,} documents")

    # Run examples
    example_1_find_related_topics(documents, embeddings, topic_id=0, top_k=5)
    example_2_temporal_trends(documents)
    example_3_subtopic_diversity(documents, subtopic_keywords)
    example_4_keyword_cooccurrence(topic_keywords)
    example_5_outlier_analysis(documents, embeddings)

    print("\n" + "="*80)
    print("CUSTOM ANALYSIS COMPLETE")
    print("="*80)
    print("\nThese examples show how to:")
    print("  • Find related topics using embeddings")
    print("  • Analyze temporal trends")
    print("  • Measure sub-topic diversity")
    print("  • Analyze keyword patterns")
    print("  • Study outlier characteristics")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)