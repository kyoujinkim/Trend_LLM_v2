"""
Visualization Module
Creates visualizations for cluster analysis and time series data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os


class ClusterVisualizer:
    """Create visualizations for cluster analysis"""

    def __init__(self, output_dir: str = './visualizations'):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.family'] = 'Malgun Gothic'  # For Korean text
        plt.rcParams['axes.unicode_minus'] = False

    def plot_clusters_2d(self,
                         embeddings: np.ndarray,
                         labels: List[int],
                         title: str = "Cluster Visualization (2D)",
                         save_path: Optional[str] = None) -> None:
        """
        Plot clusters in 2D space.

        Args:
            embeddings: 2D embeddings (n_samples, 2)
            labels: Cluster labels
            title: Plot title
            save_path: Path to save the plot
        """
        if embeddings.shape[1] > 2:
            print("Warning: Embeddings have more than 2 dimensions. Using first 2 dimensions.")
            embeddings = embeddings[:, :2]

        plt.figure(figsize=(14, 10))
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise points in black
                color = 'k'
                marker = 'x'
                alpha = 0.3
            else:
                marker = 'o'
                alpha = 0.6

            mask = np.array(labels) == label
            plt.scatter(embeddings[mask, 0],
                       embeddings[mask, 1],
                       c=[color],
                       label=f'Cluster {label}' if label != -1 else 'Noise',
                       marker=marker,
                       alpha=alpha,
                       s=50)

        plt.title(title, fontsize=16)
        plt.xlabel('Dimension 1', fontsize=12)
        plt.ylabel('Dimension 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        plt.tight_layout()

        if save_path is None:
            save_path = f'{self.output_dir}/clusters_2d.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"2D cluster plot saved to {save_path}")

    def plot_clusters_3d_interactive(self,
                                      embeddings: np.ndarray,
                                      labels: List[int],
                                      hover_text: Optional[List[str]] = None,
                                      title: str = "Interactive 3D Cluster Visualization",
                                      save_path: Optional[str] = None) -> None:
        """
        Create interactive 3D cluster visualization.

        Args:
            embeddings: 3D embeddings (n_samples, 3)
            labels: Cluster labels
            hover_text: Optional hover text for each point
            title: Plot title
            save_path: Path to save the plot
        """
        if embeddings.shape[1] < 3:
            print("Error: Embeddings must have at least 3 dimensions for 3D plot")
            return

        df = pd.DataFrame({
            'x': embeddings[:, 0],
            'y': embeddings[:, 1],
            'z': embeddings[:, 2],
            'cluster': labels
        })

        if hover_text is not None:
            df['text'] = hover_text

        fig = px.scatter_3d(
            df,
            x='x',
            y='y',
            z='z',
            color='cluster',
            hover_data=['text'] if hover_text is not None else None,
            title=title,
            labels={'cluster': 'Cluster ID'},
            color_continuous_scale='Viridis'
        )

        fig.update_traces(marker=dict(size=3))
        fig.update_layout(height=800, width=1000)

        if save_path is None:
            save_path = f'{self.output_dir}/clusters_3d_interactive.html'
        fig.write_html(save_path)
        print(f"Interactive 3D plot saved to {save_path}")

    def plot_cluster_sizes(self,
                           cluster_stats: Dict,
                           title: str = "Cluster Sizes",
                           save_path: Optional[str] = None) -> None:
        """
        Plot cluster sizes as bar chart.

        Args:
            cluster_stats: Dictionary with cluster statistics
            title: Plot title
            save_path: Path to save the plot
        """
        topics = list(cluster_stats.keys())
        sizes = [cluster_stats[t]['size'] for t in topics]
        labels = [cluster_stats[t].get('label', f'Topic {t}') for t in topics]

        # Sort by size
        sorted_indices = np.argsort(sizes)[::-1]
        topics = [topics[i] for i in sorted_indices]
        sizes = [sizes[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]

        plt.figure(figsize=(14, 8))
        bars = plt.bar(range(len(topics)), sizes, color=sns.color_palette("husl", len(topics)))
        plt.xlabel('Topic', fontsize=12)
        plt.ylabel('Number of Documents', fontsize=12)
        plt.title(title, fontsize=16)
        plt.xticks(range(len(topics)), topics, rotation=45, ha='right')

        # Add value labels on bars
        for i, (bar, size) in enumerate(zip(bars, sizes)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{int(size)}',
                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path is None:
            save_path = f'{self.output_dir}/cluster_sizes.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Cluster sizes plot saved to {save_path}")

    def plot_time_series(self,
                         time_series_data: Dict,
                         topics: Optional[List[int]] = None,
                         title: str = "Topic Trends Over Time",
                         save_path: Optional[str] = None) -> None:
        """
        Plot time series for selected topics.

        Args:
            time_series_data: Dictionary with time series data
            topics: List of topics to plot (None for all)
            title: Plot title
            save_path: Path to save the plot
        """
        if topics is None:
            topics = list(time_series_data.keys())[:10]  # Plot top 10 if not specified

        plt.figure(figsize=(16, 8))

        for topic_id in topics:
            if topic_id in time_series_data:
                data = time_series_data[topic_id]
                dates = pd.to_datetime(data['dates'])
                counts = data['counts']
                plt.plot(dates, counts, marker='o', label=f'Topic {topic_id}', linewidth=2)

        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Document Count', fontsize=12)
        plt.title(title, fontsize=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path is None:
            save_path = f'{self.output_dir}/time_series.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Time series plot saved to {save_path}")

    def plot_interactive_time_series(self,
                                      time_series_data: Dict,
                                      title: str = "Interactive Topic Trends Over Time",
                                      save_path: Optional[str] = None) -> None:
        """
        Create interactive time series visualization.

        Args:
            time_series_data: Dictionary with time series data
            title: Plot title
            save_path: Path to save the plot
        """
        fig = go.Figure()

        for topic_id, data in time_series_data.items():
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(data['dates']),
                y=data['counts'],
                mode='lines+markers',
                name=f'Topic {topic_id}',
                hovertemplate='<b>Topic %{fullData.name}</b><br>Date: %{x}<br>Count: %{y}<extra></extra>'
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Document Count',
            hovermode='x unified',
            height=600,
            width=1200,
            showlegend=True
        )

        if save_path is None:
            save_path = f'{self.output_dir}/time_series_interactive.html'
        fig.write_html(save_path)
        print(f"Interactive time series plot saved to {save_path}")

    def plot_heatmap(self,
                     time_series_data: Dict,
                     title: str = "Topic Activity Heatmap",
                     save_path: Optional[str] = None) -> None:
        """
        Create heatmap showing topic activity over time.

        Args:
            time_series_data: Dictionary with time series data
            title: Plot title
            save_path: Path to save the plot
        """
        # Prepare data for heatmap
        all_dates = []
        for data in time_series_data.values():
            all_dates.extend(data['dates'])
        unique_dates = sorted(set(all_dates))

        # Create matrix
        matrix_data = []
        topic_ids = sorted(time_series_data.keys())

        for topic_id in topic_ids:
            data = time_series_data[topic_id]
            date_to_count = dict(zip(data['dates'], data['counts']))
            row = [date_to_count.get(date, 0) for date in unique_dates]
            matrix_data.append(row)

        matrix_data = np.array(matrix_data)

        # Create heatmap
        plt.figure(figsize=(20, 10))
        sns.heatmap(matrix_data,
                   xticklabels=[d[:10] for d in unique_dates[::max(1, len(unique_dates)//20)]],
                   yticklabels=[f'Topic {t}' for t in topic_ids],
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Document Count'},
                   linewidths=0.5)

        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Topic', fontsize=12)
        plt.title(title, fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path is None:
            save_path = f'{self.output_dir}/topic_heatmap.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved to {save_path}")

    def plot_topic_network(self,
                           cluster_relationships: Dict,
                           cluster_stats: Dict,
                           title: str = "Topic Relationship Network",
                           save_path: Optional[str] = None) -> None:
        """
        Create network visualization of topic relationships.

        Args:
            cluster_relationships: Dictionary with topic relationships
            cluster_stats: Dictionary with cluster statistics
            title: Plot title
            save_path: Path to save the plot
        """
        # Prepare edges
        edge_x = []
        edge_y = []
        edge_weights = []

        # Create a simple circular layout
        topics = list(cluster_stats.keys())
        n = len(topics)
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        pos = {topic: (np.cos(t), np.sin(t)) for topic, t in zip(topics, theta)}

        for topic_i, related in cluster_relationships.items():
            if topic_i in pos:
                for rel in related:
                    topic_j = rel['related_topic']
                    if topic_j in pos:
                        edge_x.extend([pos[topic_i][0], pos[topic_j][0], None])
                        edge_y.extend([pos[topic_i][1], pos[topic_j][1], None])
                        edge_weights.append(rel['similarity'])

        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        # Create node trace
        node_x = [pos[t][0] for t in topics]
        node_y = [pos[t][1] for t in topics]
        node_sizes = [cluster_stats[t]['size'] for t in topics]
        node_text = [f"Topic {t}<br>Size: {cluster_stats[t]['size']}" for t in topics]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[f'T{t}' for t in topics],
            textposition='top center',
            hovertext=node_text,
            marker=dict(
                size=[np.sqrt(s)*2 for s in node_sizes],
                color=topics,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Topic ID",
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=title,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0, l=0, r=0, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=800,
                           width=1000
                       ))

        if save_path is None:
            save_path = f'{self.output_dir}/topic_network.html'
        fig.write_html(save_path)
        print(f"Topic network plot saved to {save_path}")

    def create_dashboard(self,
                         cluster_stats: Dict,
                         time_series_data: Dict,
                         trending_topics: List[Dict],
                         save_path: Optional[str] = None) -> None:
        """
        Create comprehensive dashboard with multiple visualizations.

        Args:
            cluster_stats: Dictionary with cluster statistics
            time_series_data: Dictionary with time series data
            trending_topics: List of trending topics
            save_path: Path to save the dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cluster Sizes', 'Time Series (Top 5 Topics)',
                          'Trending Topics', 'Topic Distribution'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )

        # 1. Cluster sizes
        topics = list(cluster_stats.keys())[:10]
        sizes = [cluster_stats[t]['size'] for t in topics]
        fig.add_trace(go.Bar(x=[f'T{t}' for t in topics], y=sizes, name='Size'),
                     row=1, col=1)

        # 2. Time series for top 5 topics
        top_topics = sorted(cluster_stats.keys(), key=lambda t: cluster_stats[t]['size'], reverse=True)[:5]
        for topic_id in top_topics:
            if topic_id in time_series_data:
                data = time_series_data[topic_id]
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(data['dates']),
                    y=data['counts'],
                    mode='lines',
                    name=f'Topic {topic_id}'
                ), row=1, col=2)

        # 3. Trending topics
        if trending_topics:
            trending_ids = [t['topic_id'] for t in trending_topics[:10]]
            trending_growth = [t['growth_percentage'] for t in trending_topics[:10]]
            fig.add_trace(go.Bar(x=[f'T{t}' for t in trending_ids],
                                y=trending_growth,
                                name='Growth %'),
                         row=2, col=1)

        # 4. Topic distribution pie chart
        fig.add_trace(go.Pie(labels=[f'Topic {t}' for t in topics],
                            values=sizes,
                            name='Distribution'),
                     row=2, col=2)

        fig.update_layout(height=1000, width=1600, title_text="Cluster Analysis Dashboard")

        if save_path is None:
            save_path = f'{self.output_dir}/dashboard.html'
        fig.write_html(save_path)
        print(f"Dashboard saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    print("Visualization module - use as import")