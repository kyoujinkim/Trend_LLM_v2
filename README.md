# Trend Discovery with LLM

A comprehensive pipeline for discovering and analyzing trending topics in Korean text datasets using embedding, clustering, and LLM techniques.

## Project Overview

This project implements an end-to-end pipeline for trend discovery in large text datasets. It uses state-of-the-art embedding models, clustering algorithms, and LLM-based analysis to identify, track, and visualize trending topics over time.

### Key Features

- **Text Embedding**: Convert Korean text to high-dimensional embeddings using sentence transformers
- **Smart Caching**: Efficient embedding storage and retrieval with mapper-based caching
- **Advanced Clustering**: UMAP dimensionality reduction + HDBSCAN clustering with sub-topic detection
- **LLM Keyword Extraction**: Automatic keyword extraction using Large Language Models
- **Cluster Analysis**: Comprehensive statistics, relationships, and cohesion metrics
- **Time-Series Tracking**: Track cluster evolution, detect merge/split events, analyze stability with FFT
- **Rich Visualizations**: 2D/3D plots, timelines, heatmaps, and interactive dashboards

## Project Structure

```
Trend_LLM_v2/
├── main.py                     # Main pipeline orchestrator
├── text_to_embedding.py        # Text to embedding conversion
├── topic_cluster.py            # Clustering with UMAP + HDBSCAN
├── llm_keyword_extractor.py    # LLM-based keyword extraction
├── cluster_analyzer.py         # Cluster statistics and relationships
├── cluster_tracker.py          # Time-series tracking and evolution
├── visualizer.py               # Visualization module
├── project_config.ini          # Project configuration
├── requirements.txt            # Python dependencies
└── data/
    ├── news/                   # Input CSV files with text data
    ├── embeddings/             # Generated embeddings (cached)
    ├── emb_mapper.json         # Embedding-to-document mapping
    └── output/                 # All analysis results
        ├── model_components.pkl
        ├── topic_keywords.json
        ├── subtopic_keywords.json
        ├── analysis/
        │   ├── cluster_statistics.json
        │   └── cluster_relationships.json
        ├── tracking/
        │   ├── cluster_timeline.json
        │   ├── evolution_events.json
        │   ├── stability_metrics.json
        │   └── merge_split_events.json
        └── visualizations/
            ├── clusters_2d.png
            ├── clusters_3d.html
            ├── cluster_sizes.png
            ├── cluster_timeline.png
            ├── cluster_heatmap.png
            └── interactive_dashboard.html
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)
- HuggingFace account and API token

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Trend_LLM_v2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the project:

Create or edit `project_config.ini`:
```ini
[embedding]
model_id = google/embeddinggemma-300M
batch_size = 64
iterate_size = 20480

[data]
data_path = ./data
```

4. Create API configuration file (e.g., `D:/config.ini`):
```ini
[huggingface]
token = your_huggingface_token_here
```

## Usage

### Basic Usage

Run the complete pipeline:
```bash
python main.py
```

### Advanced Usage

#### Run Specific Steps

```bash
# Step 1: Generate embeddings only
python main.py --step 1

# Step 2: Clustering only
python main.py --step 2

# Step 3: Keyword extraction only
python main.py --step 3

# Step 4: Cluster analysis only
python main.py --step 4

# Step 5: Evolution tracking only
python main.py --step 5

# Step 6: Visualization only
python main.py --step 6
```

#### Command-Line Options

```bash
python main.py [OPTIONS]

Options:
  --step INT              Which step to run (0=all, 1-6=specific step)
  --stdate YYYY-MM-DD     Start date for data filtering
  --enddate YYYY-MM-DD    End date for data filtering
  --force-regenerate      Force regenerate embeddings
  --min-topic-size INT    Minimum topic size for clustering (default: 10)
  --load-model           Load existing model instead of training new one
  --skip-llm             Skip LLM keyword extraction
  --skip-visualization   Skip visualization step
```

#### Examples

```bash
# Process data from specific date range
python main.py --stdate 2024-01-01 --enddate 2024-12-31

# Use larger minimum topic size
python main.py --min-topic-size 50

# Skip LLM extraction to save time/resources
python main.py --skip-llm

# Force regenerate embeddings
python main.py --step 1 --force-regenerate
```

## Pipeline Steps

### Step 1: Generate Embeddings

Converts text documents into high-dimensional embeddings using sentence transformers.

**Features:**
- Batch processing for efficiency
- Automatic caching with mapper
- Support for date range filtering
- Progress tracking with tqdm

**Outputs:**
- `data/embeddings/*.pkl` - Embedding files
- `data/emb_mapper.json` - Mapping of documents to embeddings

### Step 2: Cluster Topics

Applies UMAP dimensionality reduction and HDBSCAN clustering to discover topics.

**Features:**
- UMAP for dimensionality reduction (10D)
- HDBSCAN for density-based clustering
- Outlier re-clustering for better coverage
- Sub-topic detection within main topics
- Representative document extraction

**Outputs:**
- `data/output/model_components.pkl` - Trained models
- Topic assignments for all documents

### Step 3: Extract Keywords (LLM)

Uses Large Language Models to extract representative keywords from clusters.

**Features:**
- Gemma-2-2B model for Korean text
- Context-aware keyword extraction
- Main topic and sub-topic keywords
- Automatic GPU/CPU detection

**Outputs:**
- `data/output/topic_keywords.json`
- `data/output/subtopic_keywords.json`

### Step 4: Analyze Clusters

Calculates comprehensive statistics and relationships between clusters.

**Features:**
- Cluster cohesion and similarity metrics
- Sub-topic diversity analysis
- Time-based statistics
- Cluster relationship detection
- Source/field overlap analysis

**Outputs:**
- `data/output/analysis/cluster_statistics.json`
- `data/output/analysis/cluster_relationships.json`
- Summary report printed to console

### Step 5: Track Evolution

Tracks cluster changes over time and detects evolution events.

**Features:**
- Daily timeline tracking
- Growth/shrinkage detection
- Emergence/disappearance events
- Merge/split detection
- FFT-based stability analysis
- Periodicity detection

**Outputs:**
- `data/output/tracking/cluster_timeline.json`
- `data/output/tracking/evolution_events.json`
- `data/output/tracking/stability_metrics.json`
- `data/output/tracking/merge_split_events.json`

### Step 6: Visualize

Creates comprehensive visualizations of analysis results.

**Features:**
- 2D cluster scatter plots
- Interactive 3D visualizations
- Timeline charts
- Activity heatmaps
- Stability analysis plots
- Interactive dashboard

**Outputs:**
- `data/output/visualizations/clusters_2d.png`
- `data/output/visualizations/clusters_3d.html`
- `data/output/visualizations/cluster_timeline.png`
- `data/output/visualizations/cluster_heatmap.png`
- `data/output/visualizations/interactive_dashboard.html`

## Input Data Format

Input CSV files should be placed in `data/news/` with the following columns:

- `title`: Document title
- `content`: Document content (can be XML formatted)
- `date`: Date in YYYYMMDD format
- `time`: Time information
- `source`: Source identifier
- `kind`: Document kind/category
- `class`: Document class

## Configuration

### Embedding Configuration

```ini
[embedding]
model_id = google/embeddinggemma-300M  # HuggingFace model ID
batch_size = 64                         # Batch size for encoding
iterate_size = 20480                    # Documents per embedding file
```

### Data Configuration

```ini
[data]
data_path = ./data  # Path to data directory
```

## Performance Tips

1. **GPU Usage**: Use CUDA-capable GPU for faster embedding generation and LLM processing
2. **Batch Size**: Adjust `batch_size` based on available GPU memory
3. **Skip LLM**: Use `--skip-llm` flag to skip keyword extraction if not needed
4. **Model Caching**: Use `--load-model` to reuse previously trained clustering models
5. **Embedding Cache**: Embeddings are automatically cached and reused

## Troubleshooting

### Out of Memory Errors

- Reduce `batch_size` in configuration
- Reduce `iterate_size` in configuration
- Use `--skip-llm` to avoid loading LLM model
- Process smaller date ranges

### Korean Font Issues in Visualizations

The visualizer automatically tries to find Korean fonts. If text doesn't display correctly:
- Install Korean fonts (Malgun Gothic, NanumGothic, etc.)
- Or set `use_korean_font=False` in visualizer initialization

### Missing Dependencies

```bash
pip install -r requirements.txt --upgrade
```

## Dependencies

Key dependencies:
- `torch`: PyTorch for deep learning
- `transformers`: HuggingFace transformers
- `sentence-transformers`: Sentence embeddings
- `umap-learn`: Dimensionality reduction
- `hdbscan`: Density-based clustering
- `scikit-learn`: Machine learning utilities
- `pandas`, `numpy`: Data manipulation
- `matplotlib`, `seaborn`: Static visualizations
- `plotly`: Interactive visualizations
- `scipy`: Scientific computing (FFT, etc.)

See `requirements.txt` for complete list with versions.

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Citation

If you use this project in your research, please cite:

```
[Add citation information here]
```

## Contact

[Add contact information here]