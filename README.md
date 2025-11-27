# Trend Discovery Using LLM Embeddings

A comprehensive pipeline for discovering trending topics in large Korean text datasets using embeddings and clustering techniques.

## Project Overview

This project implements an end-to-end pipeline that:
1. Converts text into embeddings using sentence transformers
2. Reduces dimensionality using UMAP
3. Clusters documents using HDBSCAN
4. Extracts topic representations using c-TF-IDF
5. Analyzes cluster statistics and relationships
6. Tracks temporal evolution of topics
7. Creates interactive visualizations

## Features

- **Efficient Embedding Generation**: Batch processing with caching for reusability
- **Advanced Clustering**: UMAP + HDBSCAN for robust topic discovery
- **Topic Representation**: c-TF-IDF based keyword extraction
- **Temporal Analysis**: Track topic trends, detect trending/declining topics
- **Cluster Evolution**: Monitor merge, split, growth, and shrinkage of topics
- **Stability Analysis**: FFT-based periodicity detection and variance analysis
- **Rich Visualizations**: 2D/3D plots, time series, heatmaps, network graphs, and interactive dashboards

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create API configuration file at `D:/config.ini` (or specify custom path):
```ini
[huggingface]
token = your_huggingface_token_here
```

3. Prepare your data:
   - Place CSV files in `./data/` directory
   - CSV files should contain columns: `title`, `content`, `date`, `class`, `time`, `source`, `kind`
   - Content can be in XML format (will be cleaned automatically)

## Project Structure

```
Trend_LLM_v2/
├── text_to_embedding.py      # Step 1: Generate embeddings
├── topic_cluster.py           # Step 2-3: Clustering and topic extraction
├── cluster_analyzer.py        # Step 4: Cluster analysis
├── time_series_tracker.py     # Step 5: Temporal analysis
├── visualization.py           # Step 6: Visualization
├── main.py                    # Pipeline orchestrator
├── project_config.ini         # Project configuration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Usage

### Quick Start

Run the complete pipeline:

```bash
python main.py
```

### Advanced Usage

```bash
python main.py \
  --project-config ./project_config.ini \
  --api-config D:/config.ini \
  --stdate 2024-01-01 \
  --enddate 2024-12-31 \
  --min-topic-size 15 \
  --freq W \
  --force-regenerate
```

### Arguments

- `--project-config`: Path to project configuration file (default: `./project_config.ini`)
- `--api-config`: Path to API configuration file (default: `D:/config.ini`)
- `--stdate`: Start date for filtering (format: YYYY-MM-DD)
- `--enddate`: End date for filtering (format: YYYY-MM-DD)
- `--force-regenerate`: Force regenerate embeddings even if they exist
- `--min-topic-size`: Minimum documents per topic (default: 10)
- `--date-column`: Name of date column (default: `date_dt`)
- `--freq`: Time series frequency - D (daily), W (weekly), M (monthly) (default: D)

### Individual Steps

You can also run individual steps:

```python
from main import TrendDiscoveryPipeline

# Initialize pipeline
pipeline = TrendDiscoveryPipeline('./project_config.ini', 'D:/config.ini')

# Step 1: Generate embeddings
pipeline.step1_generate_embeddings()

# Step 2: Cluster topics
model, documents = pipeline.step2_cluster_topics(min_topic_size=15)

# Step 3: Analyze clusters
analyzer, stats, relationships = pipeline.step3_analyze_clusters(model, documents)

# Step 4: Temporal analysis
tracker, ts_data, trending, declining = pipeline.step4_temporal_analysis(documents)

# Step 5: Visualize
pipeline.step5_visualize(model, documents, stats, relationships, ts_data, trending)
```

## Configuration

Edit `project_config.ini` to customize:

```ini
[embedding]
model_id = google/embeddinggemma-300M
batch_size = 64
iterate_size = 20480

[data]
data_path = ./data
```

## Output

The pipeline generates comprehensive output in `./data/output/`:

### Clustering Results (`output/clustering/`)
- `topic_info.json`: Topic representations with keywords and scores
- `representative_docs.json`: Most representative documents per topic
- `topic_assignments.csv`: All documents with topic assignments
- `model_components.pkl`: Saved model components

### Analysis Results (`output/analysis/`)
- `cluster_statistics.json`: Size, percentage, confidence metrics
- `cluster_relationships.json`: Topic similarity relationships
- `topic_diversity.json`: Entropy and concentration metrics

### Temporal Analysis (`output/temporal/`)
- `time_series_data.json`: Topic frequencies over time
- `cluster_lifespans.json`: First/last appearance, active days
- `stability_metrics.json`: Variance, FFT analysis
- `trending_topics.json`: Rapidly growing topics
- `declining_topics.json`: Declining topics

### Visualizations (`output/visualizations/`)
- `clusters_2d.png`: 2D cluster plot
- `clusters_3d_interactive.html`: Interactive 3D visualization
- `cluster_sizes.png`: Topic size distribution
- `time_series.png`: Topic trends over time
- `time_series_interactive.html`: Interactive time series
- `topic_heatmap.png`: Activity heatmap
- `topic_network.html`: Topic relationship network
- `dashboard.html`: Comprehensive interactive dashboard

## Key Components

### Text2Embedding
Converts text to embeddings using HuggingFace models with efficient batch processing and caching.

### BertTopic_morph
Custom topic modeling implementation using:
- UMAP for dimensionality reduction
- HDBSCAN for clustering
- c-TF-IDF for topic representation

### ClusterAnalyzer
Analyzes:
- Cluster statistics (size, confidence, diversity)
- Inter-cluster relationships
- Topic evolution (merge, split, growth, shrinkage)

### TimeSeriesTracker
Tracks temporal patterns:
- Frequency over time
- Cluster lifespans
- Stability metrics (variance, FFT)
- Trending and declining topics

### ClusterVisualizer
Creates rich visualizations:
- Static plots (matplotlib/seaborn)
- Interactive visualizations (plotly)
- Comprehensive dashboards

## Requirements

- Python 3.8+
- pandas >= 2.3.3
- torch >= 2.9.1
- sentence-transformers >= 5.1.2
- umap-learn >= 0.5.5
- hdbscan >= 0.8.38
- scikit-learn >= 1.5.0
- matplotlib >= 3.9.0
- plotly >= 5.24.0
- scipy >= 1.13.0

## Language Support

This project is optimized for Korean text but can work with any language supported by the chosen embedding model. For Korean text visualization, the code uses 'Malgun Gothic' font.

## Performance Tips

1. **Embeddings**: Generated embeddings are cached, so subsequent runs are much faster
2. **Batch Size**: Adjust `batch_size` in config based on your GPU memory
3. **Iterate Size**: Larger `iterate_size` uses more memory but is faster
4. **Min Topic Size**: Larger values create fewer, larger topics
5. **UMAP Components**: Reduce to 2-3 for faster processing

## Troubleshooting

**Out of Memory**:
- Reduce `batch_size` in config
- Reduce `iterate_size` in config
- Set `low_memory=True` in UMAP

**No topics found**:
- Reduce `min_topic_size`
- Check that your data has sufficient variation
- Ensure text cleaning is working correctly

**Visualization errors**:
- Install Korean font: `Malgun Gothic` for Windows
- For other OS, update font in `visualization.py`

## License

This project is provided as-is for research and educational purposes.

## Citation

If you use this code in your research, please cite:
```
Trend Discovery using LLM Embeddings
https://github.com/your-repo/trend-llm-v2
```