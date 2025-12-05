"""
Quick Start Example
Demonstrates basic usage of the Trend Discovery Pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import configparser
from main import TrendDiscoveryPipeline


def quick_start_example():
    """
    Run a quick start example of the pipeline

    This example demonstrates:
    1. Loading configuration
    2. Running the full pipeline
    3. Accessing results
    """

    print("="*80)
    print("QUICK START EXAMPLE")
    print("="*80)

    # Step 1: Load configurations
    print("\n1. Loading configurations...")

    project_config = configparser.ConfigParser()
    project_config.read('./project_config.ini')

    api_config = configparser.ConfigParser()
    # Update this path to your API config location
    api_config.read('D:/config.ini')

    print("   ✓ Configurations loaded")

    # Step 2: Initialize pipeline
    print("\n2. Initializing pipeline...")
    pipeline = TrendDiscoveryPipeline(project_config, api_config)
    print("   ✓ Pipeline initialized")

    # Step 3: Generate embeddings
    print("\n3. Generating embeddings...")
    print("   This may take a while for large datasets...")

    # For this example, we'll process a date range
    # Adjust these dates based on your data
    pipeline.step1_generate_embeddings(
        stdate='2024-01-01',
        enddate='2024-01-31',
        force_regenerate=False  # Set to True to regenerate existing embeddings
    )

    # Step 4: Cluster topics
    print("\n4. Clustering topics...")
    representative_docs, documents, embeddings, umap_embeddings = pipeline.step2_cluster_topics(
        min_topic_size=10,
        save_path=os.path.join(pipeline.output_path, 'model'),
        load_model=False  # Set to True to load existing model
    )

    print(f"\n   Found {len(set(documents['Topic'])) - 1} topics")
    print(f"   Total documents: {len(documents)}")

    # Step 5: Analyze clusters
    print("\n5. Analyzing clusters...")
    cluster_stats, cluster_relationships = pipeline.step3_analyze_clusters(
        documents, embeddings
    )

    print(f"   Analyzed {len(cluster_stats)} clusters")

    # Step 6: Track evolution
    print("\n6. Tracking temporal evolution...")
    timeline, events, stability = pipeline.step4_track_evolution(
        documents, embeddings
    )

    print(f"   Detected {len(events)} evolution events")

    # Step 7: Extract keyphrases
    print("\n7. Extracting keyphrases with LLM...")
    keywords, subtopic_keywords = pipeline.step5_extract_keyphrase(
        representative_docs,
        use_llm=True
    )

    print(f"   Extracted keywords for {len(keywords)} topics")

    # Step 8: Visualize
    print("\n8. Creating visualizations...")
    visualizer = pipeline.step6_visualize(
        umap_embeddings,
        documents,
        cluster_stats,
        timeline=timeline,
        keywords=keywords
    )

    print(f"\n   Visualizations saved to: {os.path.join(pipeline.output_path, 'visualizations')}")

    # Step 9: Display results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    print(f"\nTop 5 Topics:")
    topic_counts = documents[documents['Topic'] != -1]['Topic'].value_counts()
    for rank, (topic_id, count) in enumerate(topic_counts.head(5).items(), 1):
        kw = keywords.get(str(topic_id), ['N/A'])
        print(f"{rank}. Topic {topic_id}: {count} documents")
        print(f"   Keywords: {', '.join(kw[:5])}")

    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {pipeline.output_path}")
    print("\nNext steps:")
    print("  - View visualizations in output/visualizations/")
    print("  - Explore clusters: python explore_clusters.py")
    print("  - Generate report: python analyze_results.py --summary")


if __name__ == "__main__":
    # Check if data exists
    if not os.path.exists('./data/news'):
        print("ERROR: No data found at ./data/news/")
        print("Please add your CSV files to ./data/news/ before running this example")
        sys.exit(1)

    # Check if config exists
    if not os.path.exists('./project_config.ini'):
        print("ERROR: project_config.ini not found")
        print("Run: python setup.py")
        sys.exit(1)

    try:
        quick_start_example()
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nPlease ensure:")
        print("  1. Data files are in ./data/news/")
        print("  2. API keys are configured")
        print("  3. All dependencies are installed")
        sys.exit(1)