# pip install umap-learn
import collections
import json
import os
import pickle
from glob import glob
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
from umap import UMAP
from hdbscan import approximate_predict, HDBSCAN
from sklearn.preprocessing import normalize

class BertTopic_morph():

    def __init__(self,
        language: str = "multilingual",
        top_n_words: int = 10,
        min_topic_size: int = 10,
        nr_topics: Union[int, str] = None,
        low_memory: bool = False,
        calculate_probabilities: bool = False,
        seed_topic_list: List[List[str]] = None,
        zeroshot_topic_list: List[str] = None,
        zeroshot_min_similarity: float = 0.7,
        embedding_model=None,
        save_path='./',
        load_model: bool = False,
        verbose: bool = False,
    ):
        self.language = language
        self.top_n_words = top_n_words
        self.min_topic_size = min_topic_size
        self.nr_topics = nr_topics
        self.low_memory = low_memory
        self.calculate_probabilities = calculate_probabilities
        self.seed_topic_list = seed_topic_list
        self.zeroshot_topic_list = zeroshot_topic_list
        self.zeroshot_min_similarity = zeroshot_min_similarity
        self.embedding_model = embedding_model
        self.save_path = save_path
        self.load_model = load_model
        self.verbose = verbose

        self.umap_model = UMAP(
            n_neighbors=15,
            n_components=10,
            min_dist=0.0,
            metric="cosine",
            low_memory=self.low_memory,
        )

        self.hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_topic_size, metric="euclidean", cluster_selection_method="eom", core_dist_n_jobs=-1
        )

        if self.load_model:
            model_path = os.path.join(save_path, 'model_components.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    components = pickle.load(f)
                    self.umap_model = components['umap_model']
                    self.hdbscan_model = components['hdbscan_model']
                if self.verbose:
                    print(f"Loaded model components from {model_path}")
            else:
                self.load_model = False
                if self.verbose:
                    print(f"No saved model found at {model_path}, initializing new model.")

        # Public attributes
        self.topics_ = None
        self.probabilities_ = None
        self.topic_sizes_ = None
        self.topic_mapper_ = None
        self.topic_representations_ = None
        self.topic_embeddings_ = None
        self._topic_id_to_zeroshot_topic_idx = {}
        self.custom_labels_ = None
        self.representative_images_ = None
        self.representative_docs_ = {}
        self.topic_aspects_ = {}

        # Private attributes for internal tracking purposes
        self._merged_topics = None

    @property
    def topic_labels_(self):
        """Map topic IDs to their labels.
        A label is the topic ID, along with the first four words of the topic representation, joined using '_'.
        Zeroshot topic labels come from self.zeroshot_topic_list rather than the calculated representation.

        Returns:
            topic_labels: a dict mapping a topic ID (int) to its label (str)
        """
        topic_labels = {
            key: f"{key}_" + "_".join([word[0] for word in values[:4]])
            for key, values in self.topic_representations_.items()
        }
        return topic_labels

    def _update_topic_sizes(self, documents: pd.DataFrame):
        """Update topic sizes based on a DataFrame of documents.

        Arguments:
            documents: A DataFrame containing a 'Topic' column with topic assignments.
        """
        self.topic_sizes_ = collections.Counter(documents.Topic.values.tolist())
        self.topics_ = documents.Topic.astype(int).tolist()

    def _save_representative_docs(self, documents: pd.DataFrame, umap_embeddings: np.ndarray, n_docs: int = 3):
        """Save the most representative documents per topic.

        Arguments:
            documents: DataFrame with documents and their topic assignments
            umap_embeddings: UMAP reduced embeddings
            n_docs: Number of representative docs to save per topic
        """
        for topic in set(documents.Topic):
            if topic != -1:
                # Get indices of documents in this topic
                topic_indices = documents[documents.Topic == topic].index.tolist()

                if len(topic_indices) > 0:
                    # Get embeddings for this topic
                    topic_embeddings = umap_embeddings[topic_indices]

                    # Calculate centroid
                    centroid = topic_embeddings.mean(axis=0)

                    # Find closest documents to centroid
                    distances = np.linalg.norm(topic_embeddings - centroid, axis=1)
                    closest_indices = np.argsort(distances)[:n_docs]

                    # Save representative docs
                    self.representative_docs_[topic] = [
                        documents.iloc[topic_indices[idx]].Document
                        for idx in closest_indices
                    ]

    def fit_transform(
        self,
        documents: List[str],
        embeddings: np.ndarray = None,
            ) -> Tuple[List[int], Union[np.ndarray, None]]:
        """Fit the models on a collection of documents, generate topics,
        and return the probabilities and topic per document.

        Arguments:
            documents: A list of documents to fit on
            embeddings: Pre-trained document embeddings. These can be used
                        instead of the sentence-transformer model

        Returns:
            predictions: Topic predictions for each documents
            probabilities: The probability of the assigned topic per document.
                           If `calculate_probabilities` in BERTopic is set to True, then
                           it calculates the probabilities of all topics across all documents
                           instead of only the assigned topic. This, however, slows down
                           computation and may increase memory usage.
        """
        doc_ids = range(len(documents))
        documents_df = pd.DataFrame({"Document": documents, "ID": doc_ids, "Topic": None})

        if self.load_model:
            self.umap_embeddings = self.umap_model.transform(embeddings)
            self.hdbscan_model.generate_prediction_data()
            labels, probabilities = approximate_predict(self.hdbscan_model, self.umap_embeddings)
        else:
            # UMAP dimensionality reduction
            self.umap_embeddings = self.umap_model.fit_transform(embeddings, y=None)

            # Clustering with HDBSCAN
            self.hdbscan_model.fit(self.umap_embeddings, y=None)
            labels = self.hdbscan_model.labels_
            probabilities = self.hdbscan_model.probabilities_

        documents_df["Topic"] = labels

        # Update topic sizes
        self._update_topic_sizes(documents_df)

        # Map topics based on frequency
        df = pd.DataFrame(self.topic_sizes_.items(), columns=["Old_Topic", "Size"]).sort_values("Size", ascending=False)
        df = df[df.Old_Topic != -1]
        sorted_topics = {**{-1: -1}, **dict(zip(df.Old_Topic, range(len(df))))}

        # Map documents
        documents_df.Topic = documents_df.Topic.map(sorted_topics).fillna(documents_df.Topic).astype(int)

        # Update topic sizes after remapping
        self._update_topic_sizes(documents_df)

        # Save the top 3 most representative documents per topic
        self._save_representative_docs(documents_df, self.umap_embeddings)

        documents_df_tmp = documents_df[documents_df.Topic == -1]
        documents_df_tmp_umap = self.umap_embeddings[documents_df_tmp.index.tolist()]
        if len(documents_df_tmp) > 0:
            # Cluster between the outliers to find potential sub-topics
            hdbscan_outliers = HDBSCAN(
                min_cluster_size=max(2, self.min_topic_size // 5),
                metric="euclidean",
                cluster_selection_method="eom",
                core_dist_n_jobs=-1,
            )
            hdbscan_outliers.fit(documents_df_tmp_umap, y=None)
            outlier_labels = hdbscan_outliers.labels_
            # sort outlier labels
            documents_df_tmp['Topic'] = outlier_labels


        predictions = documents_df.Topic.to_list()
        self.probabilities_ = probabilities

        if not self.load_model:
            self.save_results(self.save_path)

        return predictions

    def save_results(self, output_path: str, documents_df: pd.DataFrame = None):
        """Save clustering results to files.

        Arguments:
            output_path: Directory to save results
            documents_df: Optional dataframe with documents and topic assignments
        """
        os.makedirs(output_path, exist_ok=True)

        # Save model components
        with open(f'{output_path}/model_components.pkl', 'wb') as f:
            pickle.dump({
                'umap_model': self.umap_model,
                'hdbscan_model': self.hdbscan_model,
            }, f)

        if self.verbose:
            print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # Import Embeddings and Documents

    doc_list = glob('./data/news/*.csv')
    documents = pd.concat([pd.read_csv(f, index_col=0) for f in doc_list], axis=0)

    # since content of documents is XML format, we need to extract text only
    import re
    def extract_text(xml_string):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', xml_string)
    documents['cleaned_text'] = documents['content'].apply(extract_text)

    model = BertTopic_morph()
    result = model.fit_transform((documents['title'] + '\n' + documents['cleaned_text']).tolist(), embeddings=embeddings)
    a = 1