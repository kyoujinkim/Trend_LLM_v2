"""
Cluster Visualizer
Creates various visualizations for cluster analysis results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import seaborn as sns
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os


class ClusterVisualizer:
    """Visualize cluster analysis results"""

    def __init__(self, output_path: str = './visualizations', use_korean_font: bool = True):
        """
        Initialize visualizer

        Args:
            output_path: Directory to save visualizations
            use_korean_font: Whether to configure Korean font for matplotlib
        """
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

        # Configure Korean font for matplotlib
        if use_korean_font:
            self._configure_korean_font()

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

    def _configure_korean_font(self):
        """Configure Korean font for matplotlib"""
        try:
            # Try common Korean fonts
            korean_fonts = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'AppleGothic']
            available_fonts = [f.name for f in fm.fontManager.ttflist]

            font_name = None
            for kfont in korean_fonts:
                if kfont in available_fonts:
                    font_name = kfont
                    break

            if font_name:
                rcParams['font.family'] = font_name
                rcParams['axes.unicode_minus'] = False
                print(f"Using Korean font: {font_name}")
            else:
                print("Warning: No Korean font found. Text may not display correctly.")
        except Exception as e:
            print(f"Warning: Could not configure Korean font: {e}")

    def plot_clusters_2d(
        self,
        embeddings: np.ndarray,
        topics: np.ndarray,
        title: str = "Topic Clusters (2D)",
        save_name: str = "clusters_2d.png"
    ):
        """
        Plot clusters in 2D space using UMAP or t-SNE

        Args:
            embeddings: 2D embeddings (already reduced)
            topics: Topic assignments
            title: Plot title
            save_name: Filename to save the plot
        """
        plt.figure(figsize=(14, 10))

        # Get unique topics
        unique_topics = sorted(set(topics))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_topics)))

        # Plot each cluster
        for i, topic in enumerate(unique_topics):
            mask = topics == topic
            if topic == -1:
                # Outliers in gray
                plt.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    c='gray',
                    alpha=0.3,
                    s=20,
                    label=f'Outliers ({np.sum(mask)})'
                )
            else:
                plt.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    c=[colors[i]],
                    alpha=0.6,
                    s=30,
                    label=f'Topic {topic} ({np.sum(mask)})'
                )

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Dimension 1', fontsize=12)
        plt.ylabel('Dimension 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        plt.tight_layout()

        save_path = os.path.join(self.output_path, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"2D cluster plot saved to {save_path}")

    def plot_clusters_3d_interactive(
        self,
        embeddings: np.ndarray,
        topics: np.ndarray,
        documents: Optional[pd.DataFrame] = None,
        title: str = "Interactive 3D Topic Clusters",
        save_name: str = "clusters_3d.html"
    ):
        """
        Create interactive 3D plot of clusters

        Args:
            embeddings: 3D embeddings
            topics: Topic assignments
            documents: Optional DataFrame with document information for hover text
            title: Plot title
            save_name: Filename to save the plot
        """
        # Prepare data
        if embeddings.shape[1] < 3:
            print("Warning: Embeddings must have at least 3 dimensions for 3D plot")
            return

        df_plot = pd.DataFrame({
            'x': embeddings[:, 0],
            'y': embeddings[:, 1],
            'z': embeddings[:, 2],
            'topic': topics
        })

        # Add hover text if documents provided
        if documents is not None and len(documents) == len(topics):
            df_plot['text'] = documents['title'].fillna('') if 'title' in documents.columns else ''
        else:
            df_plot['text'] = [f'Doc {i}' for i in range(len(topics))]

        # Create figure
        fig = px.scatter_3d(
            df_plot,
            x='x',
            y='y',
            z='z',
            color='topic',
            hover_data=['text'],
            title=title,
            labels={'topic': 'Topic ID'},
            color_continuous_scale='Viridis'
        )

        fig.update_traces(marker=dict(size=3))
        fig.update_layout(
            width=1200,
            height=800,
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3'
            )
        )

        save_path = os.path.join(self.output_path, save_name)
        fig.write_html(save_path)
        print(f"3D interactive plot saved to {save_path}")

    def plot_cluster_sizes(
        self,
        cluster_stats: Dict,
        title: str = "Cluster Sizes",
        save_name: str = "cluster_sizes.png"
    ):
        """
        Plot cluster size distribution

        Args:
            cluster_stats: Dictionary with cluster statistics
            title: Plot title
            save_name: Filename to save the plot
        """
        topics = sorted(cluster_stats.keys())
        sizes = [cluster_stats[t]['size'] for t in topics]

        plt.figure(figsize=(14, 6))
        bars = plt.bar(range(len(topics)), sizes, color='steelblue', alpha=0.7)

        # Color the largest clusters differently
        sorted_indices = np.argsort(sizes)[-5:]
        for idx in sorted_indices:
            bars[idx].set_color('coral')

        plt.xlabel('Topic ID', fontsize=12)
        plt.ylabel('Number of Documents', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xticks(range(len(topics)), topics, rotation=45)
        plt.tight_layout()

        save_path = os.path.join(self.output_path, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Cluster sizes plot saved to {save_path}")

    def plot_cluster_timeline(
        self,
        timeline_data: Dict,
        topic_ids: Optional[List[int]] = None,
        title: str = "Cluster Evolution Over Time",
        save_name: str = "cluster_timeline.png"
    ):
        """
        Plot cluster activity timeline

        Args:
            timeline_data: Dictionary with timeline information
            topic_ids: Optional list of specific topics to plot (plots all if None)
            title: Plot title
            save_name: Filename to save the plot
        """
        if not timeline_data:
            print("No timeline data available")
            return

        # Select topics to plot
        if topic_ids is None:
            topic_ids = sorted(timeline_data.keys())[:10]  # Plot top 10 by default

        plt.figure(figsize=(16, 8))

        for topic in topic_ids:
            if topic not in timeline_data:
                continue

            metrics = timeline_data[topic]['daily_metrics']
            dates = [m['date'] for m in metrics]
            counts = [m['count'] for m in metrics]

            plt.plot(dates, counts, marker='o', markersize=3, label=f'Topic {topic}', alpha=0.7)

        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Number of Documents', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        save_path = os.path.join(self.output_path, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Cluster timeline plot saved to {save_path}")

    def plot_cluster_heatmap(
        self,
        timeline_data: Dict,
        title: str = "Cluster Activity Heatmap",
        save_name: str = "cluster_heatmap.png"
    ):
        """
        Create heatmap of cluster activity over time

        Args:
            timeline_data: Dictionary with timeline information
            title: Plot title
            save_name: Filename to save the plot
        """
        if not timeline_data:
            print("No timeline data available")
            return

        # Prepare data for heatmap
        topics = sorted(timeline_data.keys())

        # Get all dates
        all_dates = set()
        for topic_data in timeline_data.values():
            for metric in topic_data['daily_metrics']:
                all_dates.add(metric['date'])
        all_dates = sorted(all_dates)

        # Create matrix
        matrix = np.zeros((len(topics), len(all_dates)))

        for i, topic in enumerate(topics):
            metrics = timeline_data[topic]['daily_metrics']
            date_to_count = {m['date']: m['count'] for m in metrics}

            for j, date in enumerate(all_dates):
                matrix[i, j] = date_to_count.get(date, 0)

        # Plot heatmap
        plt.figure(figsize=(16, max(8, len(topics) * 0.3)))
        sns.heatmap(
            matrix,
            cmap='YlOrRd',
            yticklabels=[f'Topic {t}' for t in topics],
            xticklabels=all_dates,
            cbar_kws={'label': 'Document Count'}
        )

        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Topic', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()

        save_path = os.path.join(self.output_path, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Cluster heatmap saved to {save_path}")

    def plot_stability_analysis(
        self,
        stability_metrics: Dict,
        title: str = "Cluster Stability Analysis",
        save_name: str = "cluster_stability.png"
    ):
        """
        Plot cluster stability metrics

        Args:
            stability_metrics: Dictionary with stability metrics
            title: Plot title
            save_name: Filename to save the plot
        """
        if not stability_metrics:
            print("No stability metrics available")
            return

        topics = sorted(stability_metrics.keys())
        stability_scores = [stability_metrics[t]['stability_score'] for t in topics]
        cv_scores = [stability_metrics[t]['coefficient_of_variation'] for t in topics]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot stability scores
        ax1.bar(range(len(topics)), stability_scores, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Topic ID', fontsize=12)
        ax1.set_ylabel('Stability Score', fontsize=12)
        ax1.set_title('Cluster Stability Scores (Higher = More Stable)', fontsize=14)
        ax1.set_xticks(range(len(topics)))
        ax1.set_xticklabels(topics, rotation=45)

        # Plot coefficient of variation
        ax2.bar(range(len(topics)), cv_scores, color='coral', alpha=0.7)
        ax2.set_xlabel('Topic ID', fontsize=12)
        ax2.set_ylabel('Coefficient of Variation', fontsize=12)
        ax2.set_title('Activity Variation (Lower = More Stable)', fontsize=14)
        ax2.set_xticks(range(len(topics)))
        ax2.set_xticklabels(topics, rotation=45)

        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()

        save_path = os.path.join(self.output_path, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Stability analysis plot saved to {save_path}")

    def create_interactive_dashboard(
        self,
        embeddings: np.ndarray,
        topics: np.ndarray,
        cluster_stats: Dict,
        timeline_data: Optional[Dict] = None,
        keywords: Optional[Dict] = None,
        save_name: str = "interactive_dashboard.html"
    ):
        """
        Create comprehensive interactive dashboard

        Args:
            embeddings: 2D embeddings
            topics: Topic assignments
            cluster_stats: Cluster statistics
            timeline_data: Optional timeline data
            keywords: Optional keywords for each topic
            save_name: Filename to save the dashboard
        """
        # Create subplots
        if timeline_data:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Cluster Distribution', 'Cluster Sizes', 'Timeline', 'Statistics'),
                specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                       [{'type': 'scatter'}, {'type': 'bar'}]]
            )
        else:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Cluster Distribution', 'Cluster Sizes'),
                specs=[[{'type': 'scatter'}, {'type': 'bar'}]]
            )

        # 1. Scatter plot of clusters
        unique_topics = sorted(set(topics))
        for topic in unique_topics:
            mask = topics == topic
            hover_text = [f'Topic {topic}<br>Keywords: {", ".join(keywords.get(topic, ["N/A"])[:3])}'
                         if keywords and topic in keywords else f'Topic {topic}'
                         for _ in range(np.sum(mask))]

            fig.add_trace(
                go.Scatter(
                    x=embeddings[mask, 0],
                    y=embeddings[mask, 1],
                    mode='markers',
                    name=f'Topic {topic}',
                    text=hover_text,
                    marker=dict(size=5),
                    showlegend=True
                ),
                row=1, col=1
            )

        # 2. Bar chart of cluster sizes
        topic_ids = sorted(cluster_stats.keys())
        sizes = [cluster_stats[t]['size'] for t in topic_ids]

        fig.add_trace(
            go.Bar(
                x=[f'Topic {t}' for t in topic_ids],
                y=sizes,
                name='Cluster Size',
                showlegend=False
            ),
            row=1, col=2
        )

        # 3. Timeline (if available)
        if timeline_data:
            for topic in list(timeline_data.keys())[:10]:  # Top 10 topics
                metrics = timeline_data[topic]['daily_metrics']
                dates = [m['date'] for m in metrics]
                counts = [m['count'] for m in metrics]

                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=counts,
                        mode='lines+markers',
                        name=f'Topic {topic}',
                        showlegend=False
                    ),
                    row=2, col=1
                )

            # 4. Statistics comparison
            avg_similarities = [cluster_stats[t]['avg_similarity'] for t in topic_ids]

            fig.add_trace(
                go.Bar(
                    x=[f'Topic {t}' for t in topic_ids],
                    y=avg_similarities,
                    name='Avg Similarity',
                    showlegend=False
                ),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            height=800 if timeline_data else 400,
            title_text="Cluster Analysis Dashboard",
            showlegend=True
        )

        save_path = os.path.join(self.output_path, save_name)
        fig.write_html(save_path)
        print(f"Interactive dashboard saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    print("Cluster Visualizer module loaded")
    print("Use ClusterVisualizer class to create visualizations")