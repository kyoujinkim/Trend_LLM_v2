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
from topic_cluster import BertTopic_doc
from llm_keyphrase_extractor import LLMKeyphraseExtractor
from cluster_analyzer import ClusterAnalyzer
from cluster_tracker import ClusterTracker
from visualizer import ClusterVisualizer
# ignore FutureWarnings

class TrendDiscoveryPipeline:
    """Main pipeline for discovering trending topics in text data"""

    def __init__(self, project_config, api_config):
        """
        Initialize the pipeline./content/drive/MyDrive/Trend_LLM/data/embeddings

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

        print("Initializing Text2Embedding...")
        t2e = Text2Embedding(self.api_config, self.project_config, stdate, enddate, force_regenerate)

        print("Generating embeddings...")
        t2e.run()
        print("Embeddings generated successfully!")

    def step2_cluster_topics(self, min_topic_size=10, save_path='./', load_model=False):
        """
        Step 2: Cluster embeddings to discover topics.

        Args:
            min_topic_size: Minimum number of documents in a cluster
            n_gram_range: N-gram range for topic representation
        """
        print("\n" + "="*80)
        print("STEP 2: Clustering Topics")
        print("="*80)

        documents, embeddings = self._align_doc_embeddings()

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
        print(f"Initializing topic model (min_topic_size={min_topic_size})...")
        model = BertTopic_doc(
            min_topic_size=min_topic_size,
            save_path=save_path,
            load_model=load_model,
        )

        print("Fitting topic model...")
        predictions, subpredictions = model.fit_transform(doc_strings, embeddings=embeddings)

        # Add predictions to documents
        documents['Topic'] = predictions
        documents['Subtopic'] = subpredictions

        print(f"Results saved to {self.output_path}/model")

        return model.representative_docs_, documents, embeddings, model.umap_embeddings

    def step3_analyze_clusters(self, documents, embeddings):
        """
        Step 4: Analyze cluster statistics and relationships

        Args:
            documents: DataFrame with topic assignments
            embeddings: Document embeddings

        Returns:
            Tuple of (cluster_stats, cluster_relationships)
        """
        print("\n" + "="*80)
        print("STEP 3: Analyzing Clusters")
        print("="*80)

        analyzer = ClusterAnalyzer()

        # Calculate statistics
        print("Calculating cluster statistics...")
        cluster_stats = analyzer.calculate_cluster_statistics(documents, embeddings)

        # filter Topic based on project_config's clustering - max_topic_size and max_subtopic_size
        cluster_stats = analyzer.filter_clusters_by_stat(
            max_topic_size=self.project_config.getint('clustering', 'max_topic_size', fallback=10)
            , max_subtopic_size=self.project_config.getint('clustering', 'max_subtopic_size', fallback=10)
        )

        # Find relationships
        print("Finding cluster relationships...")
        cluster_relationships = analyzer.find_cluster_relationships(documents, embeddings, top_k=5)

        # Save analysis
        #analyzer.save_analysis(os.path.join(self.output_path, 'analysis'))

        # Print summary
        print(analyzer.generate_summary_report())

        return cluster_stats, cluster_relationships

    def step4_track_evolution(self, documents, embeddings):
        """
        Step 5: Track cluster evolution over time

        Args:
            documents: DataFrame with topic assignments and dates
            embeddings: Document embeddings

        Returns:
            Tuple of (timeline, events, stability)
        """
        print("\n" + "="*80)
        print("STEP 4: Tracking Cluster Evolution")
        print("="*80)

        tracker = ClusterTracker(time_window=7)

        # Create timeline
        print("Creating cluster timeline...")
        timeline = tracker.create_cluster_timeline(documents, embeddings)

        # Detect evolution events
        print("Detecting evolution events...")
        events = tracker.detect_evolution_events(documents, embeddings)

        # Analyze stability
        print("Analyzing cluster stability...")
        stability = tracker.analyze_cluster_stability(use_fft=True)

        # Detect merge/split events
        print("Detecting merge/split events...")
        merge_split = tracker.detect_merge_split_events(documents, embeddings)

        # Save results
        tracker.save_tracking_results(os.path.join(self.output_path, 'tracking'))

        # Save stability and merge/split separately
        with open(os.path.join(self.output_path, 'tracking', 'stability_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(stability, f, ensure_ascii=False, indent=2)

        with open(os.path.join(self.output_path, 'tracking', 'merge_split_events.json'), 'w', encoding='utf-8') as f:
            json.dump(merge_split, f, ensure_ascii=False, indent=2)

        print(f"Detected {len(events)} evolution events")
        print(f"Detected {len(merge_split)} merge/split events")

        return timeline, events, stability

    def step5_extract_keyphrase(self, representative_docs, use_llm=True):
        """
        Step 3: Extract representative step3_extract_keyphrases for each cluster using LLM

        Args:
            representative_docs: Fitted topic model with representative documents
            use_llm: Whether to use LLM for keyword extraction

        Returns:
            Dictionary of step3_extract_keyphrases for each topic
        """
        print("\n" + "="*80)
        print("STEP 5: Extracting keyphrases")
        print("="*80)

        if not use_llm:
            print("Skipping LLM keyphrases extraction")
            return {}

        extractor = LLMKeyphraseExtractor(self.api_config, self.project_config)

        # Extract keywords for main topics
        keywords = extractor.extract_keywords_for_all_topics(
            representative_docs,
            save_path=os.path.join(self.output_path, 'topic_keywords.json')
        )

        # Extract keywords for subtopics
        subtopic_keywords = extractor.extract_keywords_for_subtopics(
            representative_docs,
            save_path=os.path.join(self.output_path, 'subtopic_keywords.json')
        )

        # Cleanup model to free memory
        extractor.cleanup()

        print(f"Keywords extracted for {len(keywords)} topics")
        return keywords, subtopic_keywords

    def step6_visualize(self, umap_embeddings, documents, cluster_stats, timeline=None, keywords=None):
        """
        Step 6: Create visualizations

        Args:
            umap_embeddings: UMAP reduced embeddings
            documents: DataFrame with topic assignments
            cluster_stats: Cluster statistics
            timeline: Optional timeline data
            keywords: Optional keywords for each topic
        """
        print("\n" + "="*80)
        print("STEP 6: Creating Visualizations")
        print("="*80)

        visualizer = ClusterVisualizer(
            output_path=os.path.join(self.output_path, 'visualizations'),
            use_korean_font=True
        )

        topics = documents['Topic'].values

        # 2D cluster plot
        print("Creating 2D cluster plot...")
        visualizer.plot_clusters_2d(
            umap_embeddings[:, :2],
            topics,
            title="Topic Clusters (2D)",
            save_name="clusters_2d.png"
        )

        # 3D interactive plot
        if umap_embeddings.shape[1] >= 3:
            print("Creating 3D interactive cluster plot...")
            visualizer.plot_clusters_3d_interactive(
                umap_embeddings[:, :3],
                topics,
                documents=documents,
                title="Interactive 3D Topic Clusters",
                save_name="clusters_3d.html"
            )

        # Cluster sizes
        print("Creating cluster size plot...")
        visualizer.plot_cluster_sizes(cluster_stats, save_name="cluster_sizes.png")

        # Timeline plots
        if timeline:
            print("Creating timeline visualizations...")
            visualizer.plot_cluster_timeline(timeline, save_name="cluster_timeline.png")
            visualizer.plot_cluster_heatmap(timeline, save_name="cluster_heatmap.png")

        # Interactive dashboard
        print("Creating interactive dashboard...")
        visualizer.create_interactive_dashboard(
            umap_embeddings[:, :2],
            topics,
            cluster_stats,
            timeline_data=timeline,
            keywords=keywords,
            save_name="interactive_dashboard.html"
        )

        print(f"Visualizations saved to: {os.path.join(self.output_path, 'visualizations')}")

        return visualizer

    def _align_doc_embeddings(self):
        """
        Align documents with their corresponding embeddings using emb_mapper.json
        """
        print("\n" + "="*80)
        print("Aligning Documents with Embeddings")
        print("="*80)

        # Load documents
        print("Loading documents...")
        doc_list = glob(f'{self.data_path}/news/*.csv')
        if not doc_list:
            print("Warning: No CSV files found in data path. Looking for nested directories...")
            doc_list = glob(f'{self.data_path}/**/*.csv')

        documents = pd.concat([pd.read_csv(f, encoding='UTF-8-sig') for f in tqdm(doc_list, desc="Loading documents")], axis=0)

        # Matching embeddings and documents with emb_mapper
        print("Matching embeddings with documents...")
        with open(f'{self.data_path}/emb_mapper.json', 'r', encoding='utf-8') as f:
            emb_mapper = json.load(f)

        # emb_mapper consists of { documents class + date + time + source + kind: embeddings_filename + index_in_file }
        def get_embedding_index(row):
            key = f"{row['class']}_{row['date']}_{row['time']}_{row['source']}_{row['kind']}"
            if key in emb_mapper:
                file, index = emb_mapper[key].split('_')
                return file, index
            else:
                return None, None

        documents[['emb_file', 'emb_index']] = documents.apply(get_embedding_index, axis=1, result_type='expand')

        # load embeddings based on emb_file, emb_index
        print("Loading embeddings...")
        # first load all embedding files and make list which indicate embedding file name and index
        embed_list = sorted(glob(f'{self.data_path}/embeddings/*.pkl'))
        embed_dict = {}
        for f in tqdm(embed_list, desc="Loading embedding files"):
            embed_dict[os.path.basename(f)] = pd.read_pickle(f)
        # now create embeddings array
        embeddings = []
        valid_indices = []
        for idx, row in tqdm(documents.iterrows(), desc="Matching embeddings to documents", total=len(documents)):
            if pd.notna(row['emb_file']) and pd.notna(row['emb_index']):
                emb_file = row['emb_file']
                emb_index = int(row['emb_index'])
                if emb_file in embed_dict:
                    embeddings.append(embed_dict[emb_file][emb_index])
                    valid_indices.append(idx)
        embeddings = np.array(embeddings)

        return documents, embeddings


if __name__ == "__main__":
    import configparser

    project_config = configparser.ConfigParser()
    project_config.read('./project_config.ini')
    config = configparser.ConfigParser()
    config.read('D:/config.ini')

    pipeline = TrendDiscoveryPipeline(project_config, config)

    parser = argparse.ArgumentParser(description='Trend Discovery Pipeline')
    parser.add_argument('--step', type=int, default=0, help='Which step to run (0=all, 1=embeddings, 2=clustering, 3=keywords, 4=analysis, 5=tracking, 6=visualization)')
    parser.add_argument('--stdate', type=str, default=None, help='Start date for data filtering (YYYY-MM-DD)')
    parser.add_argument('--enddate', type=str, default=None, help='End date for data filtering (YYYY-MM-DD)')
    parser.add_argument('--force-regenerate', action='store_true', help='Force regenerate embeddings')
    parser.add_argument('--load-model', default=True, action='store_true', help='Load existing model instead of training new one')
    parser.add_argument('--skip-llm', action='store_true', help='Skip LLM keyword extraction')
    parser.add_argument('--skip-visualization', action='store_true', help='Skip visualization step')

    args = parser.parse_args()
    args.step = 0

    # Step 1: Generate embeddings
    if args.step == 0 or args.step == 1:
        pipeline.step1_generate_embeddings(
            stdate=args.stdate,
            enddate=args.enddate,
            force_regenerate=args.force_regenerate
        )

    # Step 2: Cluster topics
    if args.step == 0 or args.step == 2:
        representative_docs, documents, embeddings, umap_embeddings = pipeline.step2_cluster_topics(
            min_topic_size=project_config.getint('clustering', 'min_topic_size', fallback=10),
            save_path=os.path.join(pipeline.output_path, 'model'),
            load_model=args.load_model
        )
        # save output into cache (model.representative_docs: dict, documents: DataFrame, embeddings: np.array)
        with open(os.path.join(pipeline.output_path, 'cache.pkl'), 'wb') as f:
            pickle.dump({
                'representative_docs': representative_docs,
                'documents': documents,
                'embeddings': embeddings
            }, f)
    else:
        # load from cache
        with open(os.path.join(pipeline.output_path, 'cache.pkl'), 'rb') as f:
            cache = pickle.load(f)
            representative_docs = cache['representative_docs']
            documents = cache['documents']
            embeddings = cache['embeddings']


    print(f"\nClustering complete!")
    print(f"Number of topics found: {len(set(documents['Topic'])) - 1}")
    print(f"Number of outliers: {len(documents[documents['Topic'] == -1])}")

    # Step 3: Analyze clusters
    if args.step == 0 or args.step == 3:
        cluster_stats, cluster_relationships = pipeline.step3_analyze_clusters(documents, embeddings)

    # Step 4: Track evolution
    timeline = None
    if args.step == 0 or args.step == 4:
        timeline, events, stability = pipeline.step4_track_evolution(documents, embeddings)

    # Step 5: Extract keywords
    keywords = None
    if args.step == 0 or args.step == 5:
        if not args.skip_llm:
            keywords, subtopic_keywords = pipeline.step5_extract_keyphrase(representative_docs, use_llm=True)
        else:
            print("\nSkipping LLM keyword extraction (--skip-llm flag)")
            keywords = {}
    else:
        # load keywords from previous run
        try:
            with open(os.path.join(pipeline.output_path, 'topic_keywords.json'), 'r', encoding='utf-8') as f:
                keywords = json.load(f)
            with open(os.path.join(pipeline.output_path, 'subtopic_keywords.json'), 'r', encoding='utf-8') as f:
                subtopic_keywords = json.load(f)
        except FileNotFoundError:
            raise Exception("Keywords not found in output path. Please run step 3 to generate keywords.")


    # Step 6: Visualize
    if args.step == 0 or args.step == 6:
        if not args.skip_visualization:
            visualizer = pipeline.step6_visualize(
                umap_embeddings,
                documents,
                cluster_stats,
                timeline=timeline,
                keywords=keywords
            )
        else:
            print("\nSkipping visualization (--skip-visualization flag)")

    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"All results saved to: {pipeline.output_path}")
