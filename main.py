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

        if embedding_files and not force_regenerate:
            print(f"Found {len(embedding_files)} existing embedding files. Skipping generation.")
            print("Use force_regenerate=True to regenerate embeddings.")
            return

        print("Initializing Text2Embedding...")
        t2e = Text2Embedding(self.api_config, self.project_config, stdate, enddate)

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
            verbose=True
        )

        print("Fitting topic model...")
        predictions, subpredictions = model.fit_transform(doc_strings, embeddings=embeddings)

        # Add predictions to documents
        documents['Topic'] = predictions
        documents['Subtopic'] = subpredictions

        print(f"Results saved to {self.output_path}/model")

        return model, documents


if __name__ == "__main__":
    import configparser

    project_config = configparser.ConfigParser()
    project_config.read('./project_config.ini')
    config = configparser.ConfigParser()
    config.read('D:/config.ini')

