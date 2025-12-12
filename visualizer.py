"""
Comprehensive Visualization Module for Trend Discovery Pipeline
Creates interactive and static visualizations for cluster analysis results
Based on project_blueprint.txt requirements
"""

import json
import pickle
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats


class TrendVisualizer:
    """
    Comprehensive visualization tool for trend discovery results
    Implements all visualization requirements from project_blueprint.txt
    """

    def __init__(self, output_path='./data/output', use_korean_font=True):
        """
        Initialize the visualizer

        Args:
            output_path: Path to pipeline output directory
            use_korean_font: Whether to configure Korean font support
        """
        self.output_path = Path(output_path)
        self.viz_path = self.output_path / 'visualizations'
        self.viz_path.mkdir(parents=True, exist_ok=True)

        # Configure visualization settings
        self._setup_style(use_korean_font)

        # Data containers
        self.cluster_stats = None
        self.timeline = None
        self.events = None
        self.stability = None
        self.keywords = None
        self.subtopic_keywords = None

    def _setup_style(self, use_korean_font=True):
        """Configure matplotlib and seaborn styles"""
        sns.set_palette("husl")
        plt.style.use('seaborn-v0_8-darkgrid')

        if use_korean_font:
            try:
                # Try to set up Korean font
                font_list = [f.name for f in fm.fontManager.ttflist]
                korean_fonts = ['Malgun Gothic', 'NanumGothic', 'AppleGothic']

                for font in korean_fonts:
                    if font in font_list:
                        plt.rcParams['font.family'] = font
                        plt.rcParams['axes.unicode_minus'] = False
                        break
            except Exception as e:
                print(f"Warning: Could not set Korean font: {e}")

    def load_data(self):
        """Load all required data from output directory"""
        print("Loading visualization data...")

        # Load cluster statistics
        stats_path = self.output_path / 'analysis' / 'cluster_statistics.json'
        if stats_path.exists():
            with open(stats_path, 'r', encoding='utf-8') as f:
                self.cluster_stats = json.load(f)
            print(f"  ✓ Cluster stats loaded: {len(self.cluster_stats)} topics")

        # Load timeline
        timeline_path = self.output_path / 'tracking' / 'timeline.pkl'
        if timeline_path.exists():
            with open(timeline_path, 'rb') as f:
                self.timeline = pickle.load(f)
            print(f"  ✓ Timeline loaded: {len(self.timeline)} topics")

        # Load evolution events
        events_path = self.output_path / 'tracking' / 'evolution_events.pkl'
        if events_path.exists():
            with open(events_path, 'rb') as f:
                self.events = pickle.load(f)
            print(f"  ✓ Evolution events loaded: {len(self.events)} events")

        # Load stability metrics
        stability_path = self.output_path / 'tracking' / 'stability_metrics.pkl'
        if stability_path.exists():
            with open(stability_path, 'rb') as f:
                self.stability = pickle.load(f)
            print(f"  ✓ Stability metrics loaded: {len(self.stability)} topics")

        # Load keywords
        keywords_path = self.output_path / 'topic_keywords.json'
        if keywords_path.exists():
            with open(keywords_path, 'r', encoding='utf-8') as f:
                self.keywords = json.load(f)
            print(f"  ✓ Keywords loaded: {len(self.keywords)} topics")

        # Load subtopic keywords
        subtopic_keywords_path = self.output_path / 'subtopic_keywords.json'
        if subtopic_keywords_path.exists():
            with open(subtopic_keywords_path, 'r', encoding='utf-8') as f:
                self.subtopic_keywords = json.load(f)
            print(f"  ✓ Subtopic keywords loaded")

        print("✓ All data loaded successfully\n")

    # ========================================================================
    # 1. CLUSTER OVERVIEW DASHBOARD
    # ========================================================================

    def create_overview_dashboard(self, save_name='overview_dashboard.html'):
        """
        Create interactive dashboard summarizing cluster statistics
        Implements requirement 1 from project_blueprint.txt
        """
        if not self.cluster_stats or not self.timeline:
            print("Error: Missing required data for overview dashboard")
            return

        print("Creating overview dashboard...")

        # Prepare data
        topics = list(self.cluster_stats.keys())
        sizes = [self.cluster_stats[t]['size'] for t in topics]
        cohesions = [self.cluster_stats[t]['cohesion'] for t in topics]
        n_subtopics = [self.cluster_stats[t]['n_subtopics'] for t in topics]

        # Get keywords for hover text
        hover_texts = []
        for t in topics:
            kw = self.keywords.get(str(t), ['N/A'])[:3] if self.keywords else ['N/A']
            hover_texts.append(f"Topic {t}<br>Keywords: {', '.join(kw)}<br>Size: {self.cluster_stats[t]['size']}")

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Cluster Sizes',
                'Cluster Cohesion Distribution',
                'Number of Sub-topics per Cluster',
                'Daily Activity Heatmap'
            ),
            specs=[
                [{"type": "bar"}, {"type": "box"}],
                [{"type": "bar"}, {"type": "heatmap"}]
            ]
        )

        # 1. Cluster sizes bar chart
        fig.add_trace(
            go.Bar(
                x=[f"Topic {t}" for t in topics],
                y=sizes,
                name='Cluster Size',
                marker_color='lightblue',
                hovertext=hover_texts,
                hoverinfo='text'
            ),
            row=1, col=1
        )

        # 2. Cohesion distribution box plot
        fig.add_trace(
            go.Box(
                y=cohesions,
                name='Cohesion',
                marker_color='lightgreen'
            ),
            row=1, col=2
        )

        # 3. Number of subtopics bar chart
        fig.add_trace(
            go.Bar(
                x=[f"Topic {t}" for t in topics],
                y=n_subtopics,
                name='Sub-topics',
                marker_color='lightcoral'
            ),
            row=2, col=1
        )

        # 4. Activity heatmap
        activity_matrix = self._create_activity_matrix()
        if activity_matrix is not None:
            fig.add_trace(
                go.Heatmap(
                    z=activity_matrix['values'],
                    x=activity_matrix['dates'],
                    y=activity_matrix['topics'],
                    colorscale='YlOrRd',
                    name='Activity'
                ),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title_text="Cluster Overview Dashboard",
            showlegend=False,
            height=900,
            width=1600
        )

        # Save
        output_path = self.viz_path / save_name
        fig.write_html(str(output_path))
        print(f"  ✓ Overview dashboard saved: {output_path}")

        return fig

    def _create_activity_matrix(self) -> Optional[Dict]:
        """Create activity matrix for heatmap visualization"""
        if not self.timeline:
            return None

        # Collect all dates and topics
        all_dates = set()
        topic_activities = {}

        for topic_id, data in self.timeline.items():
            daily_metrics = data.get('daily_metrics', [])
            if not daily_metrics:
                continue

            topic_activities[topic_id] = {}
            for metric in daily_metrics:
                date = metric['date']
                count = metric['count']
                all_dates.add(date)
                topic_activities[topic_id][date] = count

        if not all_dates:
            return None

        # Sort dates
        sorted_dates = sorted(list(all_dates))
        sorted_topics = sorted(topic_activities.keys())

        # Create matrix
        matrix = []
        for topic_id in sorted_topics:
            row = [topic_activities[topic_id].get(date, 0) for date in sorted_dates]
            matrix.append(row)

        return {
            'values': matrix,
            'dates': sorted_dates,
            'topics': [f"T{t}" for t in sorted_topics]
        }

    # ========================================================================
    # 2. TREND ANALYSIS GRAPHS
    # ========================================================================

    def plot_trend_analysis(self, topic_ids: Optional[List[int]] = None, save_name='trend_analysis.html'):
        """
        Create detailed trend analysis graphs for selected topics
        Implements requirement 2 from project_blueprint.txt

        Args:
            topic_ids: List of topic IDs to visualize. If None, uses top 5 topics by size
        """
        if not self.timeline:
            print("Error: Timeline data not loaded")
            return

        print("Creating trend analysis graphs...")

        # Select topics if not specified
        if topic_ids is None:
            if self.cluster_stats:
                topic_sizes = {t: self.cluster_stats[t]['size'] for t in self.cluster_stats}
                topic_ids = sorted(topic_sizes, key=topic_sizes.get, reverse=True)[:5]
            else:
                topic_ids = list(self.timeline.keys())[:5]

        # Convert to integers for dict access
        topic_ids = [int(t) for t in topic_ids]

        # Create figure with subplots
        n_topics = len(topic_ids)
        fig = make_subplots(
            rows=n_topics,
            cols=1,
            subplot_titles=[
                f"Topic {tid}: {', '.join(self.keywords.get(str(tid), ['N/A'])[:3]) if self.keywords else f'Topic {tid}'}"
                for tid in topic_ids
            ],
            vertical_spacing=0.08
        )

        # Plot each topic
        for idx, topic_id in enumerate(topic_ids, 1):
            if topic_id not in self.timeline:
                continue

            metrics = self.timeline[topic_id].get('daily_metrics', [])
            if not metrics:
                continue

            dates = [m['date'] for m in metrics]
            counts = [m['count'] for m in metrics]
            drifts = [m.get('drift', 0) for m in metrics]

            # Document count line
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=counts,
                    name=f'T{topic_id} Count',
                    mode='lines+markers',
                    line=dict(width=2),
                    yaxis='y'
                ),
                row=idx, col=1
            )

            # Add drift as secondary line
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=drifts,
                    name=f'T{topic_id} Drift',
                    mode='lines',
                    line=dict(width=1, dash='dash'),
                    yaxis='y2',
                    opacity=0.6
                ),
                row=idx, col=1
            )

            # Update axes for this subplot
            fig.update_xaxes(title_text="Date", row=idx, col=1)
            fig.update_yaxes(title_text="Document Count", row=idx, col=1)

        # Update layout
        fig.update_layout(
            title_text="Trend Analysis: Daily Metrics & Drift",
            height=400 * n_topics,
            showlegend=True,
            hovermode='x unified'
        )

        # Save
        output_path = self.viz_path / save_name
        fig.write_html(str(output_path))
        print(f"  ✓ Trend analysis saved: {output_path}")

        return fig

    # ========================================================================
    # 3. EVENT HIGHLIGHTING
    # ========================================================================

    def plot_events_timeline(self, topic_id: Optional[int] = None, save_name='events_timeline.html'):
        """
        Visualize timeline with event highlighting
        Implements requirement 3 from project_blueprint.txt

        Args:
            topic_id: Specific topic to visualize. If None, shows all topics
        """
        if not self.timeline or not self.events:
            print("Error: Timeline and events data required")
            return

        print("Creating event-highlighted timeline...")

        fig = go.Figure()

        # Determine topics to plot
        topics_to_plot = [topic_id] if topic_id is not None else list(self.timeline.keys())

        for tid in topics_to_plot:
            if tid not in self.timeline:
                continue

            metrics = self.timeline[tid].get('daily_metrics', [])
            if not metrics:
                continue

            dates = [m['date'] for m in metrics]
            counts = [m['count'] for m in metrics]

            # Plot main trend line
            topic_label = ', '.join(self.keywords.get(str(tid), [f'Topic {tid}'])[:3]) if self.keywords else f'Topic {tid}'
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=counts,
                    name=topic_label,
                    mode='lines+markers',
                    line=dict(width=2)
                )
            )

            # Add event annotations
            topic_events = []
            for e in self.events:
                if isinstance(e, pd.DataFrame):
                    # Event is a DataFrame, filter by topic
                    if hasattr(e, 'index') and 'topic' in str(e):
                        topic_events.append(e)
                elif isinstance(e, dict) and e.get('topic') == tid:
                    topic_events.append(e)

            # Handle DataFrame events
            for event in topic_events:
                if isinstance(event, pd.DataFrame):
                    for date_idx in event.index:
                        event_date = str(date_idx)
                        event_count = next((m['count'] for m in metrics if m['date'] == event_date), None)

                        if event_count is not None:
                            trend_type = event.loc[date_idx, 'trend'] if 'trend' in event.columns else 'event'
                            color_map = {
                                'strong_growth': 'green',
                                'weak_growth': 'lightgreen',
                                'strong_shrinkage': 'red',
                                'weak_shrinkage': 'orange',
                                'stable': 'blue'
                            }
                            color = color_map.get(trend_type, 'gray')

                            fig.add_trace(
                                go.Scatter(
                                    x=[event_date],
                                    y=[event_count],
                                    mode='markers',
                                    marker=dict(size=15, color=color, symbol='star'),
                                    name=trend_type,
                                    showlegend=False,
                                    hovertext=f"{trend_type}<br>Date: {event_date}",
                                    hoverinfo='text'
                                )
                            )
                elif isinstance(event, dict):
                    if 'date' not in event:
                        continue

                    event_type = event.get('trend', event.get('type', 'event'))
                    event_date = event['date']
                    event_count = next((m['count'] for m in metrics if m['date'] == event_date), None)

                    if event_count is not None:
                        color_map = {
                            'strong_growth': 'green',
                            'weak_growth': 'lightgreen',
                            'strong_shrinkage': 'red',
                            'weak_shrinkage': 'orange',
                            'stable': 'blue'
                        }
                        color = color_map.get(event_type, 'gray')

                        fig.add_trace(
                            go.Scatter(
                                x=[event_date],
                                y=[event_count],
                                mode='markers',
                                marker=dict(size=15, color=color, symbol='star'),
                                name=event_type,
                                showlegend=False,
                                hovertext=f"{event_type}<br>Date: {event_date}<br>ADX: {event.get('ADX', 'N/A')}",
                                hoverinfo='text'
                            )
                        )

        # Update layout
        fig.update_layout(
            title="Timeline with Event Highlighting",
            xaxis_title="Date",
            yaxis_title="Document Count",
            hovermode='x unified',
            height=600
        )

        # Save
        output_path = self.viz_path / save_name
        fig.write_html(str(output_path))
        print(f"  ✓ Events timeline saved: {output_path}")

        return fig

    # ========================================================================
    # 4. STABILITY VISUALIZATION
    # ========================================================================

    def plot_stability_analysis(self, save_name='stability_analysis.html'):
        """
        Create stability visualizations
        Implements requirement 4 from project_blueprint.txt
        """
        if not self.stability:
            print("Error: Stability data not loaded")
            return

        print("Creating stability analysis...")

        # Prepare data
        topics = list(self.stability.keys())
        stability_scores = [self.stability[t]['stability_score'] for t in topics]
        drift_means = [self.stability[t]['drift_mean'] for t in topics]
        drift_stds = [self.stability[t]['drift_std'] for t in topics]
        cvs = [self.stability[t]['coefficient_of_variation'] for t in topics]

        # Get keywords for hover
        hover_texts = []
        for t in topics:
            kw = self.keywords.get(str(t), ['N/A'])[:3] if self.keywords else ['N/A']
            hover_texts.append(
                f"Topic {t}<br>Keywords: {', '.join(kw)}<br>"
                f"Stability: {self.stability[t]['stability_score']:.3f}<br>"
                f"Drift Mean: {self.stability[t]['drift_mean']:.3f}"
            )

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Stability Score Distribution',
                'Stability vs Drift',
                'Drift Mean vs Std',
                'Coefficient of Variation'
            ),
            specs=[
                [{"type": "histogram"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )

        # 1. Stability score histogram
        fig.add_trace(
            go.Histogram(
                x=stability_scores,
                name='Stability Score',
                nbinsx=20,
                marker_color='lightblue'
            ),
            row=1, col=1
        )

        # 2. Stability vs Drift scatter
        fig.add_trace(
            go.Scatter(
                x=drift_means,
                y=stability_scores,
                mode='markers',
                marker=dict(
                    size=10,
                    color=stability_scores,
                    colorscale='Viridis',
                    showscale=True
                ),
                text=hover_texts,
                hoverinfo='text',
                name='Topics'
            ),
            row=1, col=2
        )

        # 3. Drift mean vs std
        fig.add_trace(
            go.Scatter(
                x=drift_means,
                y=drift_stds,
                mode='markers',
                marker=dict(
                    size=10,
                    color=cvs,
                    colorscale='RdYlGn_r',
                    showscale=True
                ),
                text=hover_texts,
                hoverinfo='text',
                name='Drift Analysis'
            ),
            row=2, col=1
        )

        # 4. CV bar chart (top 10 most variable)
        sorted_indices = np.argsort(cvs)[-10:]
        top_topics = [topics[i] for i in sorted_indices]
        top_cvs = [cvs[i] for i in sorted_indices]

        fig.add_trace(
            go.Bar(
                x=[f"T{t}" for t in top_topics],
                y=top_cvs,
                name='CV',
                marker_color='lightcoral'
            ),
            row=2, col=2
        )

        # Update axes
        fig.update_xaxes(title_text="Stability Score", row=1, col=1)
        fig.update_xaxes(title_text="Drift Mean", row=1, col=2)
        fig.update_xaxes(title_text="Drift Mean", row=2, col=1)
        fig.update_xaxes(title_text="Topic", row=2, col=2)

        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Stability Score", row=1, col=2)
        fig.update_yaxes(title_text="Drift Std", row=2, col=1)
        fig.update_yaxes(title_text="Coefficient of Variation", row=2, col=2)

        # Update layout
        fig.update_layout(
            title_text="Stability Analysis Dashboard",
            showlegend=False,
            height=900,
            width=1600
        )

        # Save
        output_path = self.viz_path / save_name
        fig.write_html(str(output_path))
        print(f"  ✓ Stability analysis saved: {output_path}")

        return fig

    # ========================================================================
    # 5. INTERACTIVE FEATURES
    # ========================================================================

    def create_interactive_explorer(self, save_name='interactive_explorer.html'):
        """
        Create comprehensive interactive data explorer
        Implements requirement 5 from project_blueprint.txt
        """
        if not self.cluster_stats or not self.timeline:
            print("Error: Missing required data")
            return

        print("Creating interactive explorer...")

        # Prepare comprehensive data
        topics = []
        for tid in self.cluster_stats.keys():
            topic_data = {
                'topic_id': tid,
                'size': self.cluster_stats[tid]['size'],
                'cohesion': self.cluster_stats[tid]['cohesion'],
                'n_subtopics': self.cluster_stats[tid]['n_subtopics'],
                'keywords': ', '.join(self.keywords.get(str(tid), ['N/A'])[:3]) if self.keywords else 'N/A'
            }

            # Add stability metrics if available
            if self.stability and tid in self.stability:
                topic_data['stability_score'] = self.stability[tid]['stability_score']
                topic_data['drift_mean'] = self.stability[tid]['drift_mean']
            else:
                topic_data['stability_score'] = 0
                topic_data['drift_mean'] = 0

            # Add timeline info if available
            if self.timeline and tid in self.timeline:
                total_docs = self.timeline[tid].get('total_documents', 0)
                date_range = self.timeline[tid].get('date_range', {})
                topic_data['total_documents'] = total_docs
                topic_data['date_range'] = f"{date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}"
                topic_data['duration_days'] = date_range.get('days', 0)

            topics.append(topic_data)

        df = pd.DataFrame(topics)

        # Create interactive scatter plot with multiple dimensions
        fig = px.scatter(
            df,
            x='size',
            y='cohesion',
            size='n_subtopics',
            color='stability_score',
            hover_data=['topic_id', 'keywords', 'drift_mean', 'duration_days'],
            title='Interactive Topic Explorer',
            labels={
                'size': 'Cluster Size',
                'cohesion': 'Cohesion',
                'stability_score': 'Stability Score',
                'n_subtopics': 'Sub-topics'
            },
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            height=700,
            hovermode='closest'
        )

        # Save
        output_path = self.viz_path / save_name
        fig.write_html(str(output_path))
        print(f"  ✓ Interactive explorer saved: {output_path}")

        return fig

    # ========================================================================
    # 6. SUMMARY REPORTS
    # ========================================================================

    def generate_visual_summary_report(self, save_name='summary_report.html'):
        """
        Generate automated visual summary report
        Implements requirement 6 from project_blueprint.txt
        """
        print("Generating visual summary report...")

        # Create comprehensive report with multiple visualizations
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Top 10 Topics by Size',
                'Cluster Quality Metrics',
                'Temporal Activity Overview',
                'Stability Score Distribution',
                'Event Type Distribution',
                'Key Insights Summary'
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "table"}]
            ]
        )

        # 1. Top topics by size
        if self.cluster_stats:
            topics = list(self.cluster_stats.keys())
            sizes = [self.cluster_stats[t]['size'] for t in topics]
            sorted_indices = np.argsort(sizes)[-10:]
            top_topics = [topics[i] for i in sorted_indices]
            top_sizes = [sizes[i] for i in sorted_indices]

            labels = []
            for t in top_topics:
                kw = self.keywords.get(str(t), ['N/A'])[:2] if self.keywords else ['N/A']
                labels.append(f"T{t}: {', '.join(kw)}")

            fig.add_trace(
                go.Bar(
                    x=top_sizes,
                    y=labels,
                    orientation='h',
                    marker_color='lightblue'
                ),
                row=1, col=1
            )

        # 2. Quality metrics (cohesion vs similarity)
        if self.cluster_stats:
            topics = list(self.cluster_stats.keys())
            cohesions = [self.cluster_stats[t]['cohesion'] for t in topics]
            similarities = [self.cluster_stats[t]['avg_similarity'] for t in topics]

            fig.add_trace(
                go.Scatter(
                    x=similarities,
                    y=cohesions,
                    mode='markers',
                    marker=dict(size=8, color='lightgreen'),
                    name='Quality'
                ),
                row=1, col=2
            )

        # 3. Temporal activity
        if self.timeline:
            activity_by_date = {}
            for tid, tdata in self.timeline.items():
                for metric in tdata.get('daily_metrics', []):
                    date = metric['date']
                    count = metric['count']
                    activity_by_date[date] = activity_by_date.get(date, 0) + count

            dates = sorted(activity_by_date.keys())
            counts = [activity_by_date[d] for d in dates]

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=counts,
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='lightcoral'),
                    name='Activity'
                ),
                row=2, col=1
            )

        # 4. Stability distribution
        if self.stability:
            stability_scores = [self.stability[t]['stability_score'] for t in self.stability]

            fig.add_trace(
                go.Histogram(
                    x=stability_scores,
                    nbinsx=20,
                    marker_color='lightyellow',
                    name='Stability'
                ),
                row=2, col=2
            )

        # 5. Event type distribution
        if self.events:
            event_types = {}
            for event in self.events:
                if isinstance(event, pd.DataFrame):
                    # Count events in DataFrame
                    if 'trend' in event.columns:
                        for trend in event['trend']:
                            event_types[trend] = event_types.get(trend, 0) + 1
                elif isinstance(event, dict):
                    etype = event.get('trend', event.get('type', 'unknown'))
                    event_types[etype] = event_types.get(etype, 0) + 1

            fig.add_trace(
                go.Bar(
                    x=list(event_types.keys()),
                    y=list(event_types.values()),
                    marker_color='lightpink'
                ),
                row=3, col=1
            )

        # 6. Key insights table
        insights = self._generate_key_insights()

        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Metric', 'Value'],
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[
                        list(insights.keys()),
                        list(insights.values())
                    ],
                    fill_color='lavender',
                    align='left'
                )
            ),
            row=3, col=2
        )

        # Update layout
        fig.update_layout(
            title_text="Trend Discovery Summary Report",
            showlegend=False,
            height=1200,
            width=1600
        )

        # Save
        output_path = self.viz_path / save_name
        fig.write_html(str(output_path))
        print(f"  ✓ Summary report saved: {output_path}")

        return fig

    def _generate_key_insights(self) -> Dict[str, str]:
        """Generate key insights for summary report"""
        insights = {}

        if self.cluster_stats:
            n_topics = len(self.cluster_stats)
            total_docs = sum(self.cluster_stats[t]['size'] for t in self.cluster_stats)
            avg_size = total_docs / n_topics if n_topics > 0 else 0

            insights['Total Topics'] = str(n_topics)
            insights['Total Documents'] = f"{total_docs:,}"
            insights['Avg Topic Size'] = f"{avg_size:.1f}"

            # Find largest topic
            largest_topic = max(self.cluster_stats, key=lambda t: self.cluster_stats[t]['size'])
            insights['Largest Topic'] = f"Topic {largest_topic} ({self.cluster_stats[largest_topic]['size']} docs)"

        if self.stability:
            stability_scores = [self.stability[t]['stability_score'] for t in self.stability]
            insights['Avg Stability'] = f"{np.mean(stability_scores):.3f}"

            # Most stable topic
            most_stable = max(self.stability, key=lambda t: self.stability[t]['stability_score'])
            insights['Most Stable Topic'] = f"Topic {most_stable}"

        if self.events:
            total_events = 0
            for event in self.events:
                if isinstance(event, pd.DataFrame):
                    total_events += len(event)
                else:
                    total_events += 1
            insights['Total Events'] = str(total_events)

        if self.timeline:
            all_dates = []
            for tid, tdata in self.timeline.items():
                date_range = tdata.get('date_range', {})
                if 'start' in date_range and 'end' in date_range:
                    all_dates.extend([date_range['start'], date_range['end']])

            if all_dates:
                insights['Date Range'] = f"{min(all_dates)} to {max(all_dates)}"

        return insights

    # ========================================================================
    # MAIN VISUALIZATION PIPELINE
    # ========================================================================

    def create_all_visualizations(self):
        """
        Create all visualizations in one go
        This is the main entry point for generating all required visualizations
        """
        print("\n" + "="*80)
        print("CREATING ALL VISUALIZATIONS")
        print("="*80 + "\n")

        # Load data first
        self.load_data()

        # Create each visualization
        print("\n1. Creating Overview Dashboard...")
        self.create_overview_dashboard()

        print("\n2. Creating Trend Analysis...")
        self.plot_trend_analysis()

        print("\n3. Creating Events Timeline...")
        self.plot_events_timeline()

        print("\n4. Creating Stability Analysis...")
        self.plot_stability_analysis()

        print("\n5. Creating Interactive Explorer...")
        self.create_interactive_explorer()

        print("\n6. Creating Summary Report...")
        self.generate_visual_summary_report()

        print("\n" + "="*80)
        print("ALL VISUALIZATIONS COMPLETE")
        print("="*80)
        print(f"\nOutput directory: {self.viz_path}")


def main():
    """Main entry point for visualization script"""
    import argparse

    parser = argparse.ArgumentParser(description='Create trend discovery visualizations')
    parser.add_argument('--output-path', default='./data/output',
                       help='Path to pipeline output directory')
    parser.add_argument('--korean-font', action='store_true', default=True,
                       help='Enable Korean font support')
    parser.add_argument('--specific', type=str, choices=[
        'overview', 'trends', 'events', 'stability', 'explorer', 'summary', 'all'
    ], default='all', help='Which visualization to create')

    args = parser.parse_args()

    # Create visualizer
    visualizer = TrendVisualizer(
        output_path=args.output_path,
        use_korean_font=args.korean_font
    )

    # Load data
    visualizer.load_data()

    # Create requested visualizations
    if args.specific == 'all':
        visualizer.create_all_visualizations()
    elif args.specific == 'overview':
        visualizer.create_overview_dashboard()
    elif args.specific == 'trends':
        visualizer.plot_trend_analysis()
    elif args.specific == 'events':
        visualizer.plot_events_timeline()
    elif args.specific == 'stability':
        visualizer.plot_stability_analysis()
    elif args.specific == 'explorer':
        visualizer.create_interactive_explorer()
    elif args.specific == 'summary':
        visualizer.generate_visual_summary_report()


if __name__ == "__main__":
    main()