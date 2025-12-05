"""
Cluster Exploration Script
Interactive tool for exploring hierarchical cluster structure and results
"""

import json
import pickle
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict


class ClusterExplorer:
    """Interactive cluster exploration tool"""

    def __init__(self, output_path='./data/output'):
        self.output_path = Path(output_path)
        self.documents = None
        self.cluster_stats = None
        self.topic_keywords = None
        self.subtopic_keywords = None

        self.load_data()

    def load_data(self):
        """Load all necessary data files"""
        print("Loading data...")

        # Load documents and embeddings from cache
        cache_path = self.output_path / 'cache.pkl'
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
                self.documents = cache['documents']
                self.embeddings = cache.get('embeddings')
            print(f"  ✓ Loaded {len(self.documents)} documents")
        else:
            print("  ✗ cache.pkl not found")
            return

        # Load cluster statistics
        stats_path = self.output_path / 'analysis' / 'cluster_statistics.json'
        if stats_path.exists():
            with open(stats_path, 'r', encoding='utf-8') as f:
                self.cluster_stats = json.load(f)
            print(f"  ✓ Loaded statistics for {len(self.cluster_stats)} clusters")

        # Load topic keywords
        topic_kw_path = self.output_path / 'topic_keywords.json'
        if topic_kw_path.exists():
            with open(topic_kw_path, 'r', encoding='utf-8') as f:
                self.topic_keywords = json.load(f)
            print(f"  ✓ Loaded keywords for {len(self.topic_keywords)} topics")

        # Load subtopic keywords
        subtopic_kw_path = self.output_path / 'subtopic_keywords.json'
        if subtopic_kw_path.exists():
            with open(subtopic_kw_path, 'r', encoding='utf-8') as f:
                self.subtopic_keywords = json.load(f)
            print(f"  ✓ Loaded subtopic keywords")

        print("✓ Data loaded successfully\n")

    def show_overview(self):
        """Display overview of clustering results"""
        print("="*80)
        print("CLUSTER OVERVIEW")
        print("="*80)

        if self.documents is None:
            print("No data loaded")
            return

        # Overall statistics
        total_docs = len(self.documents)
        n_topics = len(self.documents['Topic'].unique()) - (1 if -1 in self.documents['Topic'].values else 0)
        n_outliers = len(self.documents[self.documents['Topic'] == -1])

        print(f"\nTotal documents: {total_docs:,}")
        print(f"Number of topics: {n_topics}")
        print(f"Number of outliers: {n_outliers:,} ({100*n_outliers/total_docs:.1f}%)")

        # Topic distribution
        print(f"\n{'Topic':<8} {'Size':<10} {'Sub-topics':<12} {'Keywords'}")
        print("-"*80)

        topic_counts = self.documents[self.documents['Topic'] != -1]['Topic'].value_counts().sort_index()

        for topic_id in topic_counts.index[:15]:  # Show top 15 topics
            size = topic_counts[topic_id]
            n_subtopics = len(self.documents[
                (self.documents['Topic'] == topic_id) & (self.documents['Subtopic'] != -1)
            ]['Subtopic'].unique())

            keywords = "N/A"
            if self.topic_keywords and str(topic_id) in self.topic_keywords:
                kw_list = self.topic_keywords[str(topic_id)]
                keywords = ', '.join(kw_list[:3]) if isinstance(kw_list, list) else str(kw_list)[:50]

            print(f"{topic_id:<8} {size:<10,} {n_subtopics:<12} {keywords}")

        if len(topic_counts) > 15:
            print(f"... and {len(topic_counts) - 15} more topics")

        print("="*80)

    def explore_topic(self, topic_id):
        """Explore a specific topic in detail"""
        topic_id = int(topic_id)

        print("\n" + "="*80)
        print(f"TOPIC {topic_id} - DETAILED VIEW")
        print("="*80)

        if self.documents is None:
            print("No data loaded")
            return

        # Get topic documents
        topic_docs = self.documents[self.documents['Topic'] == topic_id]

        if len(topic_docs) == 0:
            print(f"Topic {topic_id} not found")
            return

        # Basic statistics
        print(f"\nSize: {len(topic_docs):,} documents")

        # Keywords
        if self.topic_keywords and str(topic_id) in self.topic_keywords:
            print(f"Keywords: {', '.join(self.topic_keywords[str(topic_id)])}")

        # Cluster statistics
        if self.cluster_stats and str(topic_id) in self.cluster_stats:
            stats = self.cluster_stats[str(topic_id)]
            print(f"Similarity: {stats.get('avg_similarity', 0):.3f}")
            print(f"Cohesion: {stats.get('cohesion', 0):.3f}")
            print(f"Number of sub-topics: {stats.get('n_subtopics', 0)}")
            print(f"Sub-topic entropy: {stats.get('subtopic_entropy', 0):.3f}")

        # Date range
        if 'date' in topic_docs.columns:
            topic_docs_copy = topic_docs.copy()
            topic_docs_copy['date_dt'] = pd.to_datetime(topic_docs_copy['date'], format='%Y%m%d', errors='coerce')
            if topic_docs_copy['date_dt'].notna().any():
                date_range = f"{topic_docs_copy['date_dt'].min().strftime('%Y-%m-%d')} to {topic_docs_copy['date_dt'].max().strftime('%Y-%m-%d')}"
                print(f"Date range: {date_range}")

        # Sub-topics
        print("\n" + "-"*80)
        print("SUB-TOPICS")
        print("-"*80)

        subtopic_counts = topic_docs[topic_docs['Subtopic'] != -1]['Subtopic'].value_counts().sort_index()

        if len(subtopic_counts) == 0:
            print("No sub-topics (topic size < 100 or no sub-clustering)")
        else:
            print(f"\n{'Sub-topic':<12} {'Size':<10} {'Similarity':<12} {'Keywords'}")
            print("-"*80)

            for subtopic_id in subtopic_counts.index:
                size = subtopic_counts[subtopic_id]

                # Get similarity from cluster stats
                similarity = "N/A"
                if self.cluster_stats and str(topic_id) in self.cluster_stats:
                    subtopic_stats = self.cluster_stats[str(topic_id)].get('subtopic_stats', {})
                    if str(subtopic_id) in subtopic_stats:
                        similarity = f"{subtopic_stats[str(subtopic_id)]['avg_similarity']:.3f}"

                # Get keywords
                keywords = "N/A"
                if self.subtopic_keywords and str(topic_id) in self.subtopic_keywords:
                    if str(subtopic_id) in self.subtopic_keywords[str(topic_id)]:
                        kw_list = self.subtopic_keywords[str(topic_id)][str(subtopic_id)]
                        keywords = ', '.join(kw_list[:3]) if isinstance(kw_list, list) else str(kw_list)[:40]

                print(f"{subtopic_id:<12} {size:<10,} {similarity:<12} {keywords}")

        # Sample documents
        print("\n" + "-"*80)
        print("SAMPLE DOCUMENTS")
        print("-"*80)

        sample_docs = topic_docs.head(5)
        for idx, row in sample_docs.iterrows():
            title = row.get('title', 'N/A')[:70]
            subtopic = row.get('Subtopic', -1)
            print(f"\n[Subtopic {subtopic}] {title}")
            if 'content' in row:
                content = str(row['content'])[:200].replace('\n', ' ')
                print(f"  {content}...")

        print("\n" + "="*80)

    def explore_subtopic(self, topic_id, subtopic_id):
        """Explore a specific sub-topic in detail"""
        topic_id = int(topic_id)
        subtopic_id = int(subtopic_id)

        print("\n" + "="*80)
        print(f"TOPIC {topic_id}, SUB-TOPIC {subtopic_id} - DETAILED VIEW")
        print("="*80)

        if self.documents is None:
            print("No data loaded")
            return

        # Get subtopic documents
        subtopic_docs = self.documents[
            (self.documents['Topic'] == topic_id) &
            (self.documents['Subtopic'] == subtopic_id)
        ]

        if len(subtopic_docs) == 0:
            print(f"Sub-topic {subtopic_id} in Topic {topic_id} not found")
            return

        print(f"\nSize: {len(subtopic_docs):,} documents")

        # Keywords
        if self.subtopic_keywords and str(topic_id) in self.subtopic_keywords:
            if str(subtopic_id) in self.subtopic_keywords[str(topic_id)]:
                keywords = self.subtopic_keywords[str(topic_id)][str(subtopic_id)]
                print(f"Keywords: {', '.join(keywords)}")

        # Statistics
        if self.cluster_stats and str(topic_id) in self.cluster_stats:
            subtopic_stats = self.cluster_stats[str(topic_id)].get('subtopic_stats', {})
            if str(subtopic_id) in subtopic_stats:
                stats = subtopic_stats[str(subtopic_id)]
                print(f"Similarity: {stats['avg_similarity']:.3f}")

        # Sample documents
        print("\n" + "-"*80)
        print("SAMPLE DOCUMENTS")
        print("-"*80)

        sample_docs = subtopic_docs.head(10)
        for idx, row in sample_docs.iterrows():
            title = row.get('title', 'N/A')[:80]
            print(f"\n• {title}")

        print("\n" + "="*80)

    def compare_topics(self, topic_ids):
        """Compare multiple topics side by side"""
        topic_ids = [int(tid) for tid in topic_ids]

        print("\n" + "="*80)
        print(f"COMPARING TOPICS: {', '.join(map(str, topic_ids))}")
        print("="*80)

        if self.documents is None:
            print("No data loaded")
            return

        comparison_data = []

        for topic_id in topic_ids:
            topic_docs = self.documents[self.documents['Topic'] == topic_id]

            if len(topic_docs) == 0:
                continue

            data = {
                'Topic': topic_id,
                'Size': len(topic_docs),
                'Keywords': 'N/A',
                'Similarity': 'N/A',
                'Sub-topics': 0
            }

            if self.topic_keywords and str(topic_id) in self.topic_keywords:
                kw_list = self.topic_keywords[str(topic_id)]
                data['Keywords'] = ', '.join(kw_list[:3]) if isinstance(kw_list, list) else 'N/A'

            if self.cluster_stats and str(topic_id) in self.cluster_stats:
                stats = self.cluster_stats[str(topic_id)]
                data['Similarity'] = f"{stats.get('avg_similarity', 0):.3f}"
                data['Sub-topics'] = stats.get('n_subtopics', 0)

            comparison_data.append(data)

        if comparison_data:
            df = pd.DataFrame(comparison_data)
            print("\n" + df.to_string(index=False))
        else:
            print("No valid topics found")

        print("\n" + "="*80)

    def export_topic(self, topic_id, output_file):
        """Export all documents from a topic to CSV"""
        topic_id = int(topic_id)

        if self.documents is None:
            print("No data loaded")
            return

        topic_docs = self.documents[self.documents['Topic'] == topic_id]

        if len(topic_docs) == 0:
            print(f"Topic {topic_id} not found")
            return

        topic_docs.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✓ Exported {len(topic_docs)} documents to {output_file}")

    def search_keywords(self, keyword):
        """Find topics containing a keyword"""
        print(f"\nSearching for topics with keyword: '{keyword}'")
        print("-"*80)

        if not self.topic_keywords:
            print("No keywords loaded")
            return

        found = []
        for topic_id, keywords in self.topic_keywords.items():
            if isinstance(keywords, list):
                if any(keyword.lower() in kw.lower() for kw in keywords):
                    found.append((int(topic_id), keywords))

        if found:
            print(f"\nFound {len(found)} topics:\n")
            for topic_id, keywords in sorted(found):
                size = len(self.documents[self.documents['Topic'] == topic_id])
                print(f"Topic {topic_id} ({size:,} docs): {', '.join(keywords)}")
        else:
            print("No topics found with that keyword")

        print("-"*80)


def main():
    parser = argparse.ArgumentParser(description='Explore cluster hierarchy and results')
    parser.add_argument('--output-path', default='./data/output',
                       help='Path to output directory')
    parser.add_argument('--topic', type=int,
                       help='Explore specific topic')
    parser.add_argument('--subtopic', type=int,
                       help='Explore specific sub-topic (requires --topic)')
    parser.add_argument('--compare', nargs='+', type=int,
                       help='Compare multiple topics')
    parser.add_argument('--search', type=str,
                       help='Search for keyword in topics')
    parser.add_argument('--export', type=str,
                       help='Export topic to CSV (requires --topic)')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive mode')

    args = parser.parse_args()

    explorer = ClusterExplorer(args.output_path)

    if args.search:
        explorer.search_keywords(args.search)
    elif args.compare:
        explorer.compare_topics(args.compare)
    elif args.topic is not None:
        if args.subtopic is not None:
            explorer.explore_subtopic(args.topic, args.subtopic)
        elif args.export:
            explorer.export_topic(args.topic, args.export)
        else:
            explorer.explore_topic(args.topic)
    elif args.interactive:
        interactive_mode(explorer)
    else:
        explorer.show_overview()


def interactive_mode(explorer):
    """Interactive exploration mode"""
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("\nCommands:")
    print("  overview              - Show cluster overview")
    print("  topic <id>           - Explore topic")
    print("  subtopic <tid> <sid> - Explore sub-topic")
    print("  compare <id1> <id2>  - Compare topics")
    print("  search <keyword>     - Search for keyword")
    print("  export <tid> <file>  - Export topic to CSV")
    print("  quit                 - Exit")
    print("="*80 + "\n")

    while True:
        try:
            cmd = input("> ").strip().split()

            if not cmd:
                continue

            if cmd[0] == 'quit':
                break
            elif cmd[0] == 'overview':
                explorer.show_overview()
            elif cmd[0] == 'topic' and len(cmd) > 1:
                explorer.explore_topic(cmd[1])
            elif cmd[0] == 'subtopic' and len(cmd) > 2:
                explorer.explore_subtopic(cmd[1], cmd[2])
            elif cmd[0] == 'compare' and len(cmd) > 2:
                explorer.compare_topics(cmd[1:])
            elif cmd[0] == 'search' and len(cmd) > 1:
                explorer.search_keywords(' '.join(cmd[1:]))
            elif cmd[0] == 'export' and len(cmd) > 2:
                explorer.export_topic(cmd[1], cmd[2])
            else:
                print("Invalid command. Type 'quit' to exit.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
