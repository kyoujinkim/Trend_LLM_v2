# Trend Discovery Pipeline with LLM

A sophisticated trend discovery system for analyzing large-scale Korean text datasets using advanced embedding, clustering, and LLM-based keyphrase extraction techniques.

## Overview

This project implements an end-to-end pipeline for discovering and tracking trending topics in text data (primarily Korean news articles). It combines modern NLP techniques including transformer-based embeddings, UMAP dimensionality reduction, hierarchical clustering, and LLM-powered keyphrase extraction to identify, analyze, and visualize emerging trends over time.

## Key Features

- **Efficient Embedding Generation**: Converts text to high-quality embeddings using Google's EmbeddingGemma model with intelligent caching
- **Hierarchical Topic Discovery**: Identifies main topics and sub-topics using BERTopic with HDBSCAN clustering
- **Temporal Evolution Tracking**: Monitors how topics emerge, grow, merge, split, and decline over time
- **LLM-Powered Keyphrase Extraction**: Uses GPT-4 or EXAONE models to extract representative keyphrases for each topic
- **Comprehensive Analytics**: Calculates cluster statistics, relationships, and stability metrics
- **Rich Visualizations**: Generates 2D/3D plots, interactive dashboards, timelines, and heatmaps

## Pipeline Architecture

The pipeline consists of 6 sequential steps:

### Step 1: Generate Embeddings
- Converts text documents into dense vector representations
- Uses efficient batch processing with caching via `emb_mapper.json`
- Supports incremental updates to avoid reprocessing existing data
- Model: `google/embeddinggemma-300M`

### Step 2: Cluster Topics
- Reduces dimensionality using UMAP
- Clusters embeddings using BERTopic (built on HDBSCAN)
- Identifies main topics through density-based clustering
- **Hierarchical Sub-clustering**: Re-clusters documents within each main topic to discover sub-topics
  - Only topics with >100 documents are sub-clustered
  - Uses smaller `min_cluster_size` (1/5 of main topic minimum) for finer granularity
  - Each document gets both a `Topic` (main cluster) and `Subtopic` (sub-cluster) assignment
- Handles outliers and noise automatically
- Outliers (Topic=-1) are re-clustered iteratively to discover emerging patterns

### Step 3: Analyze Clusters
- Calculates cluster statistics (size, density, centroid positions)
- **Sub-cluster Analysis**: For each main topic, analyzes sub-topic characteristics
  - Calculates sub-topic size and internal similarity
  - Measures sub-topic entropy (diversity of sub-topics within main topic)
  - Identifies number of distinct sub-topics per main topic
- Finds relationships between clusters (similarity, hierarchy)
- Filters clusters and sub-clusters based on configurable thresholds:
  - `max_topic_size`: Maximum number of main topics to retain
  - `max_subtopic_size`: Maximum number of sub-topics to retain per main topic
  - Quality filtering: Removes low-quality clusters (size <10 AND similarity <0.5)
- Generates comprehensive analytical reports with hierarchical structure

### Step 4: Track Evolution
- Creates timeline of cluster appearances and changes
- Detects evolution events (emergence, growth, decline)
- Identifies merge and split events between main topics
- Analyzes stability using FFT and other metrics
- **Note**: Currently tracks main topic evolution; sub-topic temporal tracking is a planned enhancement (see ALGORITHM_IMPROVEMENTS.md)

### Step 5: Extract Keyphrases
- Uses LLM to extract representative keywords for each topic
- **Two-Level Keyphrase Extraction**:
  - Main topic keyphrases: From representative documents of entire cluster
  - Sub-topic keyphrases: From representative documents of each sub-cluster
- Supports multiple LLM backends (OpenAI GPT-4, EXAONE)
- Saves hierarchical keyphrase structure for visualization

### Step 6: Visualize
- 2D/3D cluster visualizations
- Interactive HTML dashboards
- Cluster size distributions
- Timeline plots and heatmaps
- Korean font support for labels

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for faster processing)

### Dependencies
```bash
pip install -r requirements.txt
```

Key packages:
- `torch` - Deep learning framework
- `transformers` - Hugging Face transformers library
- `umap-learn` - Dimensionality reduction
- `hdbscan` - Density-based clustering
- `bertopic` - Topic modeling
- `pandas`, `numpy` - Data manipulation
- `matplotlib`, `plotly` - Visualization
- `openai` - LLM API access

## Configuration

### 1. Project Configuration (`project_config.ini`)

```ini
[embedding]
model_id = google/embeddinggemma-300M
batch_size = 64
iterate_size = 20480

[data]
data_path = ./data
news_path = ./data/news

[clustering]
min_topic_size = 10          # Minimum documents per topic
max_topic_size = 10          # Maximum topics to retain
max_subtopic_size = 5        # Maximum subtopics per topic

[keyphraseextract]
model_class = openai
model_id = gpt-4.1-nano
max_keyphrase = 5            # Maximum keyphrases per topic
max_new_tokens = 250
```

### 2. API Configuration (`D:/config.ini`)

Create a separate config file for API credentials:
```ini
[huggingface]
token = your_hf_token_here

[openai]
api_key = your_openai_key_here
```

### 3. Data Structure

Organize your data as follows:
```
data/
├── news/                    # Input CSV files with news articles
│   ├── news_2024_01.csv
│   └── news_2024_02.csv
├── embeddings/              # Generated embeddings (auto-created)
├── emb_mapper.json          # Embedding index mapping (auto-created)
└── output/                  # Pipeline results (auto-created)
    ├── model/               # Trained models
    ├── analysis/            # Cluster analysis results
    ├── tracking/            # Evolution tracking data
    ├── visualizations/      # Generated plots
    ├── topic_keywords.json
    └── subtopic_keywords.json
```

### Expected CSV Format

Input CSV files should contain:
- `class`: Document category
- `date`: Publication date
- `time`: Publication time
- `source`: News source
- `kind`: Article type
- `title`: Article title
- `content`: Article content (may include XML tags)

## Quick Start

### 1. Setup

```bash
# Run setup script to install dependencies and create directory structure
python setup.py

# Or manually install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Edit the API configuration file created by setup:
```bash
# File location: ~/trend_pipeline_api_config.ini (or custom location)
# Add your HuggingFace token and OpenAI API key
```

### 3. Add Data

Place your CSV files in `data/news/` directory.

### 4. Run Pipeline

```bash
# Run complete pipeline
python main.py

# Or use the quick start example
python examples/quick_start.py
```

## Usage

### Basic Usage (Run All Steps)

```bash
python main.py
```

### Run Specific Steps

```bash
# Run only embedding generation
python main.py --step 1

# Run only clustering
python main.py --step 2

# Run analysis
python main.py --step 3

# Run evolution tracking
python main.py --step 4

# Run keyphrase extraction
python main.py --step 5

# Run visualization
python main.py --step 6
```

### Advanced Options

```bash
# Filter data by date range
python main.py --stdate 2024-01-01 --enddate 2024-12-31

# Force regenerate embeddings
python main.py --force-regenerate

# Load existing model instead of training new one
python main.py --load-model

# Skip LLM keyword extraction (faster, but less descriptive)
python main.py --skip-llm

# Skip visualization
python main.py --skip-visualization
```

### Analysis and Exploration

After running the pipeline, use these scripts to explore results:

```bash
# Generate comprehensive summary report
python analyze_results.py --summary

# Generate topic-specific report
python analyze_results.py --topic 0

# Save report to file
python analyze_results.py --summary --save report.txt

# Export CSV summary
python analyze_results.py --csv

# Explore clusters interactively
python explore_clusters.py

# View specific topic
python explore_clusters.py --topic 0

# View sub-topic
python explore_clusters.py --topic 0 --subtopic 1

# Compare multiple topics
python explore_clusters.py --compare 0 1 2

# Search for keyword
python explore_clusters.py --search "artificial intelligence"

# Export topic to CSV
python explore_clusters.py --topic 0 --export topic_0.csv

# Interactive mode
python explore_clusters.py --interactive
```

### Example Scripts

The `examples/` directory contains sample scripts:

```bash
# Quick start guide
python examples/quick_start.py

# Incremental update workflow
python examples/incremental_update.py

# Custom analysis examples
python examples/custom_analysis.py
```

## Output Files

### Model Files
- `output/model/` - BERTopic model files
- `output/cache.pkl` - Cached intermediate results

### Analysis Results
- `output/topic_keywords.json` - Main topic keyphrases
- `output/subtopic_keywords.json` - Subtopic keyphrases
- `output/tracking/timeline_data.json` - Temporal evolution data
- `output/tracking/stability_metrics.json` - Cluster stability analysis
- `output/tracking/merge_split_events.json` - Detected merge/split events

### Visualizations
- `output/visualizations/clusters_2d.png` - 2D cluster plot
- `output/visualizations/clusters_3d.html` - Interactive 3D visualization
- `output/visualizations/cluster_sizes.png` - Cluster size distribution
- `output/visualizations/cluster_timeline.png` - Temporal evolution
- `output/visualizations/cluster_heatmap.png` - Cluster activity heatmap
- `output/visualizations/interactive_dashboard.html` - Complete interactive dashboard

## Algorithm Details

### Embedding Strategy
Uses Google's EmbeddingGemma model to convert text into 300-dimensional embeddings. The mapper system creates unique keys from document metadata to enable efficient caching and prevent redundant computation.

### Clustering Approach
1. **UMAP Reduction**: Reduces embedding dimensionality while preserving local structure
2. **HDBSCAN Clustering**: Density-based clustering that automatically determines number of clusters
3. **Hierarchical Re-clustering**: Re-clusters within each main topic to discover sub-topics
4. **Outlier Handling**: Documents that don't fit any cluster are marked as outliers (Topic -1)

### Evolution Tracking
- **Time Windows**: Analyzes clusters in rolling time windows (default: 7 days)
- **Event Detection**: Identifies when clusters emerge, grow, shrink, merge, or split
- **Stability Analysis**: Uses FFT to analyze periodic patterns and stability
- **Centroid Tracking**: Monitors cluster centroid movement in embedding space

### Keyphrase Extraction
Uses LLM with carefully crafted prompts to extract meaningful keyphrases from representative documents. The LLM considers:
- Document content and context
- Temporal relevance
- Semantic coherence
- Conciseness and clarity

## Hierarchical Cluster Structure

### Understanding Main Topics and Sub-Topics

The pipeline creates a **two-level hierarchical structure**:

1. **Main Topics (Level 1)**: Broad thematic clusters discovered by HDBSCAN
   - Each document receives a `Topic` ID (0, 1, 2, ... or -1 for outliers)
   - Represents overarching themes in the dataset

2. **Sub-Topics (Level 2)**: Finer-grained clusters within each main topic
   - Each document also receives a `Subtopic` ID within its main topic
   - Represents specific aspects or nuances within the broader theme
   - Only created for topics with >100 documents

### Example Hierarchical Structure

```
Topic 0: "Technology and AI" (500 documents)
├── Subtopic 0: "ChatGPT and LLMs" (150 docs)
├── Subtopic 1: "AI Regulation" (120 docs)
├── Subtopic 2: "Computer Vision" (100 docs)
└── Subtopic -1: "Outliers within Topic 0" (130 docs)

Topic 1: "Economic Policy" (450 documents)
├── Subtopic 0: "Interest Rates" (180 docs)
├── Subtopic 1: "Inflation" (150 docs)
└── Subtopic -1: "Outliers within Topic 1" (120 docs)

Topic -1: "Outliers" (200 documents)
└── Subtopic -1: "No sub-clustering" (200 docs)
```

### Interpreting Sub-Cluster Data

#### 1. Cluster Statistics Output

The `cluster_statistics.json` file contains hierarchical data:

```json
{
  "0": {
    "size": 500,
    "avg_similarity": 0.75,
    "n_subtopics": 3,
    "subtopic_entropy": 1.05,
    "subtopic_stats": {
      "0": {"size": 150, "avg_similarity": 0.82},
      "1": {"size": 120, "avg_similarity": 0.78},
      "2": {"size": 100, "avg_similarity": 0.80}
    }
  }
}
```

**Key Metrics**:
- `n_subtopics`: Number of distinct sub-clusters (excluding outliers)
- `subtopic_entropy`: Diversity of sub-topics (higher = more diverse distribution)
  - Low entropy (~0): Most documents in one sub-topic
  - High entropy (>2): Documents evenly distributed across sub-topics
- `subtopic_stats`: Per-subtopic statistics
  - `size`: Number of documents in sub-topic
  - `avg_similarity`: Internal coherence of sub-topic (higher = tighter cluster)

#### 2. Keyphrase Hierarchy

The keyphrase extraction creates a hierarchical structure:

**`topic_keywords.json`** - Main topic keywords:
```json
{
  "0": ["artificial intelligence", "machine learning", "technology"],
  "1": ["economic policy", "central bank", "monetary policy"]
}
```

**`subtopic_keywords.json`** - Sub-topic keywords:
```json
{
  "0": {
    "0": ["ChatGPT", "large language models", "GPT-4"],
    "1": ["AI regulation", "ethics", "governance"],
    "2": ["computer vision", "image recognition", "autonomous vehicles"]
  },
  "1": {
    "0": ["interest rates", "federal reserve", "rate hikes"],
    "1": ["inflation", "consumer prices", "CPI"]
  }
}
```

#### 3. Document-Level Assignment

Each document in your DataFrame has two columns:

```python
# Example document assignments
doc_1: Topic=0, Subtopic=0  # Main topic 0, sub-topic 0
doc_2: Topic=0, Subtopic=1  # Main topic 0, sub-topic 1
doc_3: Topic=0, Subtopic=-1 # Main topic 0, outlier within topic
doc_4: Topic=-1, Subtopic=-1 # Global outlier
```

### Sub-Cluster Quality Filtering

The `filter_clusters_by_stat()` method applies hierarchical filtering:

1. **Main Topic Filtering**:
   - Removes topics with `size < 10 AND avg_similarity < 0.5`
   - Keeps only top `max_topic_size` topics (default: 10)

2. **Sub-Topic Filtering** (within each main topic):
   - Keeps only sub-topics with `size > 10`
   - Sorts by `avg_similarity` (descending)
   - Keeps only top `max_subtopic_size` sub-topics (default: 5)

### Working with Sub-Cluster Data

#### Accessing Sub-Clusters Programmatically

```python
import json
import pandas as pd

# Load cluster statistics
with open('output/analysis/cluster_statistics.json', 'r') as f:
    stats = json.load(f)

# Get all sub-topics for Topic 0
topic_0_subtopics = stats['0']['subtopic_stats']

# Iterate through all topics and their sub-topics
for topic_id, topic_info in stats.items():
    print(f"Topic {topic_id}: {topic_info['size']} documents, {topic_info['n_subtopics']} sub-topics")

    for subtopic_id, subtopic_info in topic_info['subtopic_stats'].items():
        print(f"  Sub-topic {subtopic_id}: {subtopic_info['size']} docs, sim={subtopic_info['avg_similarity']:.2f}")

# Load documents with assignments
documents = pd.read_pickle('output/cache.pkl')['documents']

# Filter documents by topic and subtopic
topic_0_subtopic_1 = documents[(documents['Topic'] == 0) & (documents['Subtopic'] == 1)]
```

### Current Limitations and Future Work

**Current Sub-Cluster Capabilities**:
- ✅ Hierarchical clustering and assignment
- ✅ Sub-cluster statistics and quality metrics
- ✅ Separate keyphrase extraction for sub-topics
- ✅ Filtering and quality control

**Planned Enhancements** (see `ALGORITHM_IMPROVEMENTS.md`):
- ⏳ **Sub-Topic Temporal Tracking**: Track how sub-topics evolve over time
- ⏳ **Sub-Topic Merge/Split Detection**: Identify when sub-topics merge or split
- ⏳ **Multi-Level Hierarchy**: Support >2 levels of clustering
- ⏳ **Sub-Topic Relationships**: Analyze relationships between sub-topics across main topics
- ⏳ **Sub-Topic Visualizations**: Dedicated visualizations for hierarchical structure

## Performance Considerations

### Memory Management
- Embeddings are loaded on-demand to reduce memory footprint
- Models are cleaned up after use
- Large arrays use memory-mapped files when possible

### Processing Speed
- Batch processing for embeddings (configurable batch size)
- Parallel processing where applicable
- Incremental updates to avoid reprocessing

### Scalability
- Designed for datasets with 10K-1M+ documents
- UMAP and HDBSCAN scale well to large datasets
- Caching prevents redundant computation

## Troubleshooting

### Common Issues

**1. Out of Memory Errors**
- Reduce `batch_size` in config
- Process data in smaller date ranges
- Use smaller embedding model

**2. Too Many/Few Topics**
- Adjust `min_topic_size` (higher = fewer topics)
- Modify UMAP parameters in `topic_cluster.py`
- Filter outliers more aggressively

**3. Poor Keyword Quality**
- Use more powerful LLM model
- Increase `max_new_tokens`
- Ensure representative documents are diverse

**4. Slow Processing**
- Enable GPU acceleration
- Increase `batch_size` if memory allows
- Use `--load-model` to skip retraining

## Project Structure

### Core Modules

- **`main.py`**: Main pipeline orchestrator
- **`text_to_embedding.py`**: Text-to-embedding conversion with caching
- **`topic_cluster.py`**: BERTopic clustering implementation with hierarchical sub-clustering
- **`cluster_analyzer.py`**: Cluster statistics and relationship analysis
- **`cluster_tracker.py`**: Temporal evolution tracking
- **`llm_keyphrase_extractor.py`**: LLM-based keyphrase extraction
- **`visualizer.py`**: Visualization generation

### Utility Scripts

- **`setup.py`**: Installation and setup automation
- **`analyze_results.py`**: Generate comprehensive analysis reports
- **`explore_clusters.py`**: Interactive cluster exploration tool
- **`requirements.txt`**: Python package dependencies

### Examples

- **`examples/quick_start.py`**: Complete pipeline walkthrough
- **`examples/incremental_update.py`**: Add new data without reprocessing
- **`examples/custom_analysis.py`**: Custom analysis templates

## Future Enhancements

See `ALGORITHM_IMPROVEMENTS.md` for detailed algorithm improvement proposals including:
- Adaptive clustering parameters
- Ensemble methods
- Real-time incremental updates
- Cross-lingual support
- Enhanced temporal weighting
- Improved outlier handling

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style conventions
- New features include appropriate documentation
- Complex algorithms include explanatory comments

## License

[Specify your license here]

## Citation

If you use this pipeline in your research, please cite:
```
[Add citation information]
```

## Contact

[Add contact information]

## Acknowledgments

- Built with BERTopic, UMAP, and HDBSCAN
- Powered by Google EmbeddingGemma and OpenAI GPT models
- Inspired by modern topic modeling and trend analysis research