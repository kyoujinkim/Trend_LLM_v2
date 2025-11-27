"""
Main Pipeline Script
Orchestrates the entire trend discovery pipeline from embeddings to visualizations
"""

import argparse
import configparser
import os
import pickle
import json
from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

from text_to_embedding import Text2Embedding
from topic_cluster import BertTopic_morph
from cluster_analyzer import ClusterAnalyzer
from time_series_tracker import TimeSeriesTracker
from visualization import ClusterVisualizer


class TrendDiscoveryPipeline:
    """Main pipeline for discovering trending topics in text data"""

    def __init__(self, project_config, api_config):
        """
        Initialize the pipeline.

        Args:
            project_config_path: Path to project configuration file
            api_config_path: Path to API configuration file (with HuggingFace token)
        """
        self.project_config = project_config

        self.api_config = api_config

        self.data_path = self.project_config.get('data', 'data_path')
        self.output_path = os.path.join(self.data_path, 'output')
        os.makedirs(self.output_path, exist_ok=True)

    def step1_generate_embeddings(self, stdate=None, enddate=None, force_regenerate=False):
        """
        Step 1: Generate embeddings for text data.

        Args:
            stdate: Start date for filtering data
            enddate: End date for filtering data
            force_regenerate: Whether to regenerate embeddings even if they exist
        """
        print("\n" + "="*80)
        print("STEP 1: Generating Embeddings")
        print("="*80)

        embedding_files = glob(f'{self.data_path}/embeddings/*.pkl')

        if embedding_files and not force_regenerate:
            print(f"Found {len(embedding_files)} existing embedding files. Skipping generation.")
            print("Use force_regenerate=True to regenerate embeddings.")
            return

        print("Initializing Text2Embedding...")
        t2e = Text2Embedding(self.api_config, self.project_config, stdate, enddate)

        print("Generating embeddings...")
        t2e.run()
        print("Embeddings generated successfully!")

    def step2_cluster_topics(self, min_topic_size=10, n_gram_range=(1, 2)):
        """
        Step 2: Cluster embeddings to discover topics.

        Args:
            min_topic_size: Minimum number of documents in a cluster
            n_gram_range: N-gram range for topic representation
        """
        print("\n" + "="*80)
        print("STEP 2: Clustering Topics")
        print("="*80)

        # Load embeddings
        print("Loading embeddings...")
        embed_list = sorted(glob(f'{self.data_path}/embeddings/*.pkl'))
        embeddings = np.concatenate([pd.read_pickle(f) for f in tqdm(embed_list, desc="Loading embeddings")], axis=0)

        # Load documents
        print("Loading documents...")
        doc_list = glob(f'{self.data_path}/*.csv')
        if not doc_list:
            print("Warning: No CSV files found in data path. Looking for nested directories...")
            doc_list = glob(f'{self.data_path}/**/*.csv')

        documents = pd.concat([pd.read_csv(f, encoding='UTF-8-sig') for f in tqdm(doc_list, desc="Loading documents")], axis=0)

        # Clean text
        print("Cleaning text...")
        import re
        def extract_text(xml_string):
            if pd.isna(xml_string):
                return ""
            clean = re.compile('<.*?>')
            return re.sub(clean, '', str(xml_string))

        documents['cleaned_text'] = documents['content'].apply(extract_text)

        # Create document strings
        doc_strings = (documents['title'].fillna('') + '\n' + documents['cleaned_text']).tolist()

        # Initialize and fit topic model
        print(f"Initializing topic model (min_topic_size={min_topic_size}, n_gram_range={n_gram_range})...")
        model = BertTopic_morph(
            min_topic_size=min_topic_size,
            n_gram_range=n_gram_range,
            verbose=True
        )

        print("Fitting topic model...")
        predictions = model.fit_transform(doc_strings, embeddings=embeddings)

        # Add predictions to documents
        documents['Topic'] = predictions

        # Save results
        print("Saving clustering results...")
        model.save_results(f'{self.output_path}/clustering', documents)

        # Save documents with topic assignments
        documents.to_csv(f'{self.output_path}/documents_with_topics.csv', index=False, encoding='utf-8-sig')

        print(f"Discovered {len(model.topic_representations_)} topics!")
        print(f"Results saved to {self.output_path}/clustering")

        return model, documents

    def step3_analyze_clusters(self, model, documents):
        """
        Step 3: Analyze cluster statistics and relationships.

        Args:
            model: Fitted topic model
            documents: DataFrame with documents and topic assignments
        """
        print("\n" + "="*80)
        print("STEP 3: Analyzing Clusters")
        print("="*80)

        analyzer = ClusterAnalyzer()

        print("Calculating cluster statistics...")
        cluster_stats = analyzer.calculate_cluster_statistics(model, documents)

        print("Finding cluster relationships...")
        cluster_relationships = analyzer.find_cluster_relationships(model)

        print("Calculating topic diversity...")
        diversity = analyzer.calculate_topic_diversity(model)

        # Save analysis
        print("Saving analysis results...")
        analyzer.save_analysis(f'{self.output_path}/analysis')

        # Save diversity metrics
        with open(f'{self.output_path}/analysis/topic_diversity.json', 'w', encoding='utf-8') as f:
            json.dump(diversity, f, ensure_ascii=False, indent=2)

        print(f"Analysis complete! Found {len(cluster_relationships)} topics with relationships.")
        print(f"Results saved to {self.output_path}/analysis")

        return analyzer, cluster_stats, cluster_relationships

    def step4_temporal_analysis(self, documents, date_column='date_dt', freq='D'):
        """
        Step 4: Perform temporal analysis on clusters.

        Args:
            documents: DataFrame with documents and topic assignments
            date_column: Name of the date column
            freq: Frequency for time series aggregation
        """
        print("\n" + "="*80)
        print("STEP 4: Temporal Analysis")
        print("="*80)

        tracker = TimeSeriesTracker()

        print("Tracking clusters over time...")
        time_series_data = tracker.track_cluster_over_time(documents, date_column, freq)

        print("Calculating cluster lifespans...")
        lifespans = tracker.calculate_cluster_lifespan(documents, date_column)

        print("Analyzing cluster stability...")
        stability = tracker.analyze_all_stabilities(method='variance')

        print("Detecting trending topics...")
        trending = tracker.detect_trending_topics()

        print("Detecting declining topics...")
        declining = tracker.detect_declining_topics()

        # Save temporal analysis
        print("Saving temporal analysis results...")
        tracker.save_time_series_data(f'{self.output_path}/temporal')

        print(f"Temporal analysis complete!")
        print(f"Found {len(trending)} trending topics and {len(declining)} declining topics.")
        print(f"Results saved to {self.output_path}/temporal")

        return tracker, time_series_data, trending, declining

    def step5_visualize(self, model, documents, cluster_stats, cluster_relationships,
                        time_series_data, trending):
        """
        Step 5: Create visualizations.

        Args:
            model: Fitted topic model
            documents: DataFrame with documents and topic assignments
            cluster_stats: Cluster statistics
            cluster_relationships: Cluster relationships
            time_series_data: Time series data
            trending: Trending topics
        """
        print("\n" + "="*80)
        print("STEP 5: Creating Visualizations")
        print("="*80)

        viz_path = f'{self.output_path}/visualizations'
        visualizer = ClusterVisualizer(output_dir=viz_path)

        # Get UMAP embeddings for visualization
        print("Generating 2D UMAP embeddings for visualization...")
        from umap import UMAP
        umap_2d = UMAP(n_components=2, random_state=42)

        # Load full embeddings
        embed_list = sorted(glob(f'{self.data_path}/embeddings/*.pkl'))
        embeddings = np.concatenate([pd.read_pickle(f) for f in tqdm(embed_list, desc="Loading embeddings")], axis=0)

        embeddings_2d = umap_2d.fit_transform(embeddings)

        print("Creating 2D cluster visualization...")
        visualizer.plot_clusters_2d(embeddings_2d, documents['Topic'].tolist())

        print("Creating 3D interactive visualization...")
        umap_3d = UMAP(n_components=3, random_state=42)
        embeddings_3d = umap_3d.fit_transform(embeddings)
        visualizer.plot_clusters_3d_interactive(embeddings_3d, documents['Topic'].tolist(),
                                                 hover_text=documents['title'].tolist()[:len(embeddings_3d)])

        print("Creating cluster size visualization...")
        visualizer.plot_cluster_sizes(cluster_stats)

        print("Creating time series visualizations...")
        visualizer.plot_time_series(time_series_data)
        visualizer.plot_interactive_time_series(time_series_data)

        print("Creating topic heatmap...")
        visualizer.plot_heatmap(time_series_data)

        print("Creating topic network visualization...")
        visualizer.plot_topic_network(cluster_relationships, cluster_stats)

        print("Creating comprehensive dashboard...")
        visualizer.create_dashboard(cluster_stats, time_series_data, trending)

        print(f"Visualizations complete! Saved to {viz_path}")

    def run_full_pipeline(self, stdate=None, enddate=None, force_regenerate_embeddings=False,
                          min_topic_size=10, n_gram_range=(1, 2), date_column='date_dt', freq='D'):
        """
        Run the complete pipeline from start to finish.

        Args:
            stdate: Start date for filtering data
            enddate: End date for filtering data
            force_regenerate_embeddings: Whether to regenerate embeddings
            min_topic_size: Minimum cluster size
            n_gram_range: N-gram range for topic representation
            date_column: Date column name
            freq: Time series frequency
        """
        print("\n" + "#"*80)
        print("# TREND DISCOVERY PIPELINE")
        print("#"*80)

        try:
            # Step 1: Generate embeddings
            self.step1_generate_embeddings(stdate, enddate, force_regenerate_embeddings)

            # Step 2: Cluster topics
            model, documents = self.step2_cluster_topics(min_topic_size, n_gram_range)

            # Step 3: Analyze clusters
            analyzer, cluster_stats, cluster_relationships = self.step3_analyze_clusters(model, documents)

            # Step 4: Temporal analysis
            tracker, time_series_data, trending, declining = self.step4_temporal_analysis(
                documents, date_column, freq)

            # Step 5: Visualizations
            self.step5_visualize(model, documents, cluster_stats, cluster_relationships,
                                time_series_data, trending)

            print("\n" + "#"*80)
            print("# PIPELINE COMPLETE!")
            print("#"*80)
            print(f"\nAll results saved to: {self.output_path}")
            print("\nSummary:")
            print(f"  - Total documents: {len(documents)}")
            print(f"  - Topics discovered: {len(model.topic_representations_)}")
            print(f"  - Trending topics: {len(trending)}")
            print(f"  - Declining topics: {len(declining)}")

        except Exception as e:
            print(f"\n ERROR: Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description='Trend Discovery Pipeline')
    parser.add_argument('--project-config', type=str, default='./project_config.ini',
                       help='Path to project configuration file')
    parser.add_argument('--api-config', type=str, default='D:/config.ini',
                       help='Path to API configuration file')
    parser.add_argument('--stdate', type=str, default=None,
                       help='Start date for data filtering (YYYY-MM-DD)')
    parser.add_argument('--enddate', type=str, default=None,
                       help='End date for data filtering (YYYY-MM-DD)')
    parser.add_argument('--force-regenerate', action='store_true',
                       help='Force regenerate embeddings')
    parser.add_argument('--min-topic-size', type=int, default=10,
                       help='Minimum topic size for clustering')
    parser.add_argument('--date-column', type=str, default='date_dt',
                       help='Name of date column in data')
    parser.add_argument('--freq', type=str, default='D',
                       help='Time series frequency (D=daily, W=weekly, M=monthly)')

    args = parser.parse_args()

    # Initialize and run pipeline
    pipeline = TrendDiscoveryPipeline(args.project_config, args.api_config)
    pipeline.run_full_pipeline(
        stdate=args.stdate,
        enddate=args.enddate,
        force_regenerate_embeddings=args.force_regenerate,
        min_topic_size=args.min_topic_size,
        date_column=args.date_column,
        freq=args.freq
    )


if __name__ == "__main__":
    main()