"""
Incremental Update Example
Demonstrates how to add new data without reprocessing everything
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import configparser
from main import TrendDiscoveryPipeline


def incremental_update_example():
    """
    Demonstrate incremental update workflow

    This example shows how to:
    1. Process new data only
    2. Update existing clusters
    3. Regenerate only necessary outputs
    """

    print("="*80)
    print("INCREMENTAL UPDATE EXAMPLE")
    print("="*80)

    # Load configurations
    project_config = configparser.ConfigParser()
    project_config.read('./project_config.ini')

    api_config = configparser.ConfigParser()
    api_config.read('D:/config.ini')

    pipeline = TrendDiscoveryPipeline(project_config, api_config)

    # Scenario: You already processed data from 2024-01-01 to 2024-01-31
    # Now you want to add 2024-02-01 to 2024-02-28

    print("\n" + "-"*80)
    print("Scenario: Adding new month of data to existing analysis")
    print("-"*80)

    # Step 1: Generate embeddings for new data only
    print("\n1. Processing new data (2024-02-01 to 2024-02-28)...")
    print("   Existing embeddings will be reused automatically")

    pipeline.step1_generate_embeddings(
        stdate='2024-02-01',
        enddate='2024-02-28',
        force_regenerate=False  # Important: Don't regenerate existing data
    )

    # Step 2: Re-cluster with ALL data (old + new)
    print("\n2. Re-clustering with combined data...")
    print("   Loading existing model and updating with new documents...")

    representative_docs, documents, embeddings, umap_embeddings = pipeline.step2_cluster_topics(
        min_topic_size=10,
        save_path=os.path.join(pipeline.output_path, 'model'),
        load_model=True  # Load existing model if available
    )

    print(f"   Total documents now: {len(documents):,}")

    # Step 3: Re-analyze clusters
    print("\n3. Updating cluster analysis...")
    cluster_stats, cluster_relationships = pipeline.step3_analyze_clusters(
        documents, embeddings
    )

    # Step 4: Update temporal tracking
    print("\n4. Updating temporal evolution tracking...")
    timeline, events, stability = pipeline.step4_track_evolution(
        documents, embeddings
    )

    new_events = [e for e in events if '2024-02' in e.get('date', '')]
    print(f"   New evolution events detected: {len(new_events)}")

    # Step 5: Update keywords (only for new or changed topics)
    print("\n5. Updating keyphrases...")
    keywords, subtopic_keywords = pipeline.step5_extract_keyphrase(
        representative_docs,
        use_llm=True
    )

    # Step 6: Regenerate visualizations
    print("\n6. Updating visualizations...")
    pipeline.step6_visualize(
        umap_embeddings,
        documents,
        cluster_stats,
        timeline=timeline,
        keywords=keywords
    )

    print("\n" + "="*80)
    print("INCREMENTAL UPDATE COMPLETE!")
    print("="*80)
    print("\nBest Practices for Incremental Updates:")
    print("  • Process new data with force_regenerate=False")
    print("  • Use load_model=True when re-clustering")
    print("  • Monitor for significant topic changes")
    print("  • Consider re-running full pipeline monthly for accuracy")


if __name__ == "__main__":
    try:
        incremental_update_example()
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)