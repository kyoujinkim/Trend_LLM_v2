"""
Results Analysis and Reporting Script
Generate comprehensive reports and analyses of pipeline results
"""

import json
import pickle
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict


class ResultsAnalyzer:
    """Analyze and generate reports from pipeline results"""

    def __init__(self, output_path='./data/output'):
        self.output_path = Path(output_path)
        self.data = {}
        self.load_all_data()

    def load_all_data(self):
        """Load all available result files"""
        print("Loading results...")

        # Load cache
        cache_path = self.output_path / 'cache.pkl'
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
                self.data['documents'] = cache.get('documents')
                self.data['embeddings'] = cache.get('embeddings')
                self.data['representative_docs'] = cache.get('representative_docs')
            print(f"  ✓ Documents: {len(self.data['documents'])}")

        # Load cluster statistics
        stats_path = self.output_path / 'analysis' / 'cluster_statistics.json'
        if stats_path.exists():
            with open(stats_path, 'r', encoding='utf-8') as f:
                self.data['cluster_stats'] = json.load(f)

        # Load keywords
        for kw_type in ['topic_keywords', 'subtopic_keywords']:
            kw_path = self.output_path / f'{kw_type}.json'
            if kw_path.exists():
                with open(kw_path, 'r', encoding='utf-8') as f:
                    self.data[kw_type] = json.load(f)

        # Load tracking data
        tracking_path = self.output_path / 'tracking'
        if tracking_path.exists():
            for file in ['cluster_timeline.json', 'evolution_events.json',
                        'stability_metrics.json', 'merge_split_events.json']:
                file_path = tracking_path / file
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.data[file.replace('.json', '')] = json.load(f)

        print("✓ Data loaded\n")

    def generate_summary_report(self, output_file=None):
        """Generate comprehensive summary report"""
        report = []
        report.append("="*80)
        report.append("TREND DISCOVERY PIPELINE - SUMMARY REPORT")
        report.append("="*80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # === DATASET OVERVIEW ===
        report.append("\n" + "="*80)
        report.append("1. DATASET OVERVIEW")
        report.append("="*80)

        if 'documents' in self.data:
            docs = self.data['documents']
            total_docs = len(docs)

            report.append(f"\nTotal documents: {total_docs:,}")

            # Date range
            if 'date' in docs.columns:
                docs_copy = docs.copy()
                docs_copy['date_dt'] = pd.to_datetime(docs_copy['date'], format='%Y%m%d', errors='coerce')
                if docs_copy['date_dt'].notna().any():
                    date_min = docs_copy['date_dt'].min().strftime('%Y-%m-%d')
                    date_max = docs_copy['date_dt'].max().strftime('%Y-%m-%d')
                    days = (docs_copy['date_dt'].max() - docs_copy['date_dt'].min()).days
                    report.append(f"Date range: {date_min} to {date_max} ({days} days)")

            # Source distribution
            if 'source' in docs.columns:
                n_sources = docs['source'].nunique()
                report.append(f"Number of sources: {n_sources}")

        # === CLUSTERING RESULTS ===
        report.append("\n" + "="*80)
        report.append("2. CLUSTERING RESULTS")
        report.append("="*80)

        if 'documents' in self.data:
            docs = self.data['documents']

            # Topic distribution
            n_topics = len(docs['Topic'].unique()) - (1 if -1 in docs['Topic'].values else 0)
            n_outliers = len(docs[docs['Topic'] == -1])
            outlier_pct = 100 * n_outliers / len(docs)

            report.append(f"\nNumber of main topics: {n_topics}")
            report.append(f"Number of outliers: {n_outliers:,} ({outlier_pct:.1f}%)")

            # Topic sizes
            topic_sizes = docs[docs['Topic'] != -1]['Topic'].value_counts().values
            report.append(f"\nTopic sizes:")
            report.append(f"  Mean: {topic_sizes.mean():.1f}")
            report.append(f"  Median: {np.median(topic_sizes):.1f}")
            report.append(f"  Min: {topic_sizes.min()}")
            report.append(f"  Max: {topic_sizes.max()}")

            # Sub-topic statistics
            total_subtopics = 0
            for topic_id in docs[docs['Topic'] != -1]['Topic'].unique():
                topic_docs = docs[docs['Topic'] == topic_id]
                n_subs = len(topic_docs[topic_docs['Subtopic'] != -1]['Subtopic'].unique())
                total_subtopics += n_subs

            report.append(f"\nTotal sub-topics: {total_subtopics}")
            if n_topics > 0:
                report.append(f"Average sub-topics per topic: {total_subtopics / n_topics:.1f}")

        # === CLUSTER QUALITY ===
        report.append("\n" + "="*80)
        report.append("3. CLUSTER QUALITY METRICS")
        report.append("="*80)

        if 'cluster_stats' in self.data:
            stats = self.data['cluster_stats']

            similarities = [s.get('avg_similarity', 0) for s in stats.values()]
            cohesions = [s.get('cohesion', 0) for s in stats.values()]

            report.append(f"\nAverage similarity across topics:")
            report.append(f"  Mean: {np.mean(similarities):.3f}")
            report.append(f"  Std: {np.std(similarities):.3f}")

            report.append(f"\nAverage cohesion across topics:")
            report.append(f"  Mean: {np.mean(cohesions):.3f}")
            report.append(f"  Std: {np.std(cohesions):.3f}")

            # Sub-topic quality
            sub_similarities = []
            for topic_stats in stats.values():
                for sub_stats in topic_stats.get('subtopic_stats', {}).values():
                    sub_similarities.append(sub_stats.get('avg_similarity', 0))

            if sub_similarities:
                report.append(f"\nSub-topic similarity:")
                report.append(f"  Mean: {np.mean(sub_similarities):.3f}")
                report.append(f"  Std: {np.std(sub_similarities):.3f}")

        # === TOP TOPICS ===
        report.append("\n" + "="*80)
        report.append("4. TOP TOPICS BY SIZE")
        report.append("="*80)

        if 'documents' in self.data and 'topic_keywords' in self.data:
            docs = self.data['documents']
            keywords = self.data['topic_keywords']

            topic_counts = docs[docs['Topic'] != -1]['Topic'].value_counts()

            report.append(f"\n{'Rank':<6} {'Topic':<8} {'Size':<10} {'Keywords'}")
            report.append("-"*80)

            for rank, (topic_id, size) in enumerate(topic_counts.head(10).items(), 1):
                kw = keywords.get(str(topic_id), [])
                kw_str = ', '.join(kw[:3]) if isinstance(kw, list) else 'N/A'
                report.append(f"{rank:<6} {topic_id:<8} {size:<10,} {kw_str}")

        # === TEMPORAL EVOLUTION ===
        if 'evolution_events' in self.data:
            report.append("\n" + "="*80)
            report.append("5. TEMPORAL EVOLUTION")
            report.append("="*80)

            events = self.data['evolution_events']

            event_types = defaultdict(int)
            for event in events:
                event_types[event['type']] += 1

            report.append(f"\nTotal evolution events: {len(events)}")
            for event_type, count in sorted(event_types.items()):
                report.append(f"  {event_type}: {count}")

        # === STABILITY METRICS ===
        if 'stability_metrics' in self.data:
            report.append("\n" + "="*80)
            report.append("6. CLUSTER STABILITY")
            report.append("="*80)

            stability = self.data['stability_metrics']

            stability_scores = [s.get('stability_score', 0) for s in stability.values()]

            report.append(f"\nStability scores:")
            report.append(f"  Mean: {np.mean(stability_scores):.3f}")
            report.append(f"  Median: {np.median(stability_scores):.3f}")

            # Most stable topics
            sorted_stability = sorted(
                stability.items(),
                key=lambda x: x[1].get('stability_score', 0),
                reverse=True
            )

            report.append(f"\nMost stable topics:")
            for topic_id, metrics in sorted_stability[:5]:
                score = metrics.get('stability_score', 0)
                cv = metrics.get('coefficient_of_variation', 0)
                report.append(f"  Topic {topic_id}: score={score:.3f}, CV={cv:.3f}")

        # === MERGE/SPLIT EVENTS ===
        if 'merge_split_events' in self.data:
            report.append("\n" + "="*80)
            report.append("7. MERGE/SPLIT EVENTS")
            report.append("="*80)

            events = self.data['merge_split_events']

            merges = [e for e in events if e['type'] == 'potential_merge']
            splits = [e for e in events if e['type'] == 'potential_split']

            report.append(f"\nPotential merge events: {len(merges)}")
            report.append(f"Potential split events: {len(splits)}")

        # === RECOMMENDATIONS ===
        report.append("\n" + "="*80)
        report.append("8. RECOMMENDATIONS")
        report.append("="*80)

        recommendations = []

        if 'documents' in self.data:
            docs = self.data['documents']
            outlier_pct = 100 * len(docs[docs['Topic'] == -1]) / len(docs)

            if outlier_pct > 30:
                recommendations.append(
                    f"• High outlier rate ({outlier_pct:.1f}%). Consider:\n"
                    "  - Lowering min_topic_size parameter\n"
                    "  - Adjusting UMAP parameters for better separation"
                )

            if 'cluster_stats' in self.data:
                similarities = [s.get('avg_similarity', 0) for s in self.data['cluster_stats'].values()]
                if np.mean(similarities) < 0.5:
                    recommendations.append(
                        "• Low average cluster similarity. Consider:\n"
                        "  - Increasing min_topic_size for tighter clusters\n"
                        "  - Reviewing outliers for potential new topics"
                    )

        if recommendations:
            report.append("\n" + "\n\n".join(recommendations))
        else:
            report.append("\n✓ No major issues detected")

        # === FOOTER ===
        report.append("\n" + "="*80)
        report.append("END OF REPORT")
        report.append("="*80)

        # Output report
        report_text = "\n".join(report)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"✓ Report saved to {output_file}")
        else:
            print(report_text)

        return report_text

    def generate_topic_report(self, topic_id, output_file=None):
        """Generate detailed report for a specific topic"""
        topic_id = int(topic_id)

        report = []
        report.append("="*80)
        report.append(f"TOPIC {topic_id} - DETAILED REPORT")
        report.append("="*80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        if 'documents' not in self.data:
            report.append("ERROR: No data loaded")
            return "\n".join(report)

        docs = self.data['documents']
        topic_docs = docs[docs['Topic'] == topic_id]

        if len(topic_docs) == 0:
            report.append(f"ERROR: Topic {topic_id} not found")
            return "\n".join(report)

        # Basic info
        report.append(f"Size: {len(topic_docs):,} documents")

        # Keywords
        if 'topic_keywords' in self.data and str(topic_id) in self.data['topic_keywords']:
            keywords = self.data['topic_keywords'][str(topic_id)]
            report.append(f"Keywords: {', '.join(keywords)}")

        # Statistics
        if 'cluster_stats' in self.data and str(topic_id) in self.data['cluster_stats']:
            stats = self.data['cluster_stats'][str(topic_id)]
            report.append(f"\nStatistics:")
            report.append(f"  Similarity: {stats.get('avg_similarity', 0):.3f}")
            report.append(f"  Cohesion: {stats.get('cohesion', 0):.3f}")
            report.append(f"  Number of sub-topics: {stats.get('n_subtopics', 0)}")
            report.append(f"  Sub-topic entropy: {stats.get('subtopic_entropy', 0):.3f}")

        # Temporal analysis
        if 'date' in topic_docs.columns:
            topic_docs_copy = topic_docs.copy()
            topic_docs_copy['date_dt'] = pd.to_datetime(topic_docs_copy['date'], format='%Y%m%d', errors='coerce')

            if topic_docs_copy['date_dt'].notna().any():
                report.append(f"\nTemporal Information:")
                report.append(f"  First appearance: {topic_docs_copy['date_dt'].min().strftime('%Y-%m-%d')}")
                report.append(f"  Last appearance: {topic_docs_copy['date_dt'].max().strftime('%Y-%m-%d')}")
                report.append(f"  Lifespan: {(topic_docs_copy['date_dt'].max() - topic_docs_copy['date_dt'].min()).days} days")

                # Activity over time
                daily_counts = topic_docs_copy.groupby(topic_docs_copy['date_dt'].dt.date).size()
                report.append(f"  Average daily documents: {daily_counts.mean():.1f}")
                report.append(f"  Peak daily documents: {daily_counts.max()}")

        # Sub-topics
        report.append("\n" + "-"*80)
        report.append("SUB-TOPICS")
        report.append("-"*80)

        subtopic_counts = topic_docs[topic_docs['Subtopic'] != -1]['Subtopic'].value_counts().sort_index()

        if len(subtopic_counts) > 0:
            for subtopic_id, size in subtopic_counts.items():
                report.append(f"\nSub-topic {subtopic_id}: {size:,} documents")

                # Keywords
                if 'subtopic_keywords' in self.data:
                    if str(topic_id) in self.data['subtopic_keywords']:
                        if str(subtopic_id) in self.data['subtopic_keywords'][str(topic_id)]:
                            kw = self.data['subtopic_keywords'][str(topic_id)][str(subtopic_id)]
                            report.append(f"  Keywords: {', '.join(kw)}")

        report.append("\n" + "="*80)

        # Output
        report_text = "\n".join(report)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"✓ Report saved to {output_file}")
        else:
            print(report_text)

        return report_text

    def generate_csv_summary(self, output_file='cluster_summary.csv'):
        """Generate CSV summary of all topics"""
        if 'documents' not in self.data:
            print("ERROR: No data loaded")
            return

        docs = self.data['documents']
        topic_counts = docs[docs['Topic'] != -1]['Topic'].value_counts()

        summary_data = []

        for topic_id in topic_counts.index:
            row = {
                'topic_id': topic_id,
                'size': topic_counts[topic_id],
                'keywords': '',
                'similarity': None,
                'n_subtopics': 0
            }

            # Keywords
            if 'topic_keywords' in self.data and str(topic_id) in self.data['topic_keywords']:
                kw = self.data['topic_keywords'][str(topic_id)]
                row['keywords'] = ', '.join(kw) if isinstance(kw, list) else ''

            # Statistics
            if 'cluster_stats' in self.data and str(topic_id) in self.data['cluster_stats']:
                stats = self.data['cluster_stats'][str(topic_id)]
                row['similarity'] = stats.get('avg_similarity')
                row['n_subtopics'] = stats.get('n_subtopics', 0)

            summary_data.append(row)

        df = pd.DataFrame(summary_data)
        df = df.sort_values('size', ascending=False)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')

        print(f"✓ CSV summary saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze pipeline results')
    parser.add_argument('--output-path', default='./data/output',
                       help='Path to output directory')
    parser.add_argument('--summary', action='store_true',
                       help='Generate summary report')
    parser.add_argument('--topic', type=int,
                       help='Generate detailed topic report')
    parser.add_argument('--csv', action='store_true',
                       help='Generate CSV summary')
    parser.add_argument('--save', type=str,
                       help='Save report to file')

    args = parser.parse_args()

    analyzer = ResultsAnalyzer(args.output_path)

    if args.topic is not None:
        analyzer.generate_topic_report(args.topic, args.save)
    elif args.csv:
        output_file = args.save if args.save else 'cluster_summary.csv'
        analyzer.generate_csv_summary(output_file)
    else:
        # Default: generate summary report
        analyzer.generate_summary_report(args.save)


if __name__ == "__main__":
    main()