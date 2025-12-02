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

class BertTopic_doc():
    """BERTopic-like topic modeling with UMAP and HDBSCAN, including sub-topic parsing."""


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
                    self.representative_docs_[topic] = {}
                    self.representative_docs_[topic]['text'] = [
                        documents.iloc[topic_indices[idx]].Document
                        for idx in closest_indices
                    ]

                    # Save subdoc's representative docs
                    subdoc_dict = {}
                    subtopics = documents[documents.Topic == topic].SubTopic.unique()
                    for subtopic in subtopics:
                        if subtopic != -1:
                            subtopic_indices = documents[(documents.Topic == topic) & (documents.SubTopic == subtopic)].index.tolist()
                            if len(subtopic_indices) > 10:
                                subtopic_embeddings = umap_embeddings[subtopic_indices]
                                sub_centroid = subtopic_embeddings.mean(axis=0)
                                sub_distances = np.linalg.norm(subtopic_embeddings - sub_centroid, axis=1)
                                sub_closest_indices = np.argsort(sub_distances)[:n_docs]
                                subdoc_dict[subtopic] = [
                                    documents.iloc[subtopic_indices[idx]].Document
                                    for idx in sub_closest_indices
                                ]

                    self.representative_docs_[topic]['subdocs'] = subdoc_dict

    def fit_transform(
        self,
        documents: List[str],
        embeddings: np.ndarray = None,
            ) -> Tuple[List[int], List[int]]:
        """Fit the models on a collection of documents, generate topics,
        and return the probabilities and topic per document.

        Arguments:
            documents: A list of documents to fit on
            embeddings: Pre-trained document embeddings. These can be used
                        instead of the sentence-transformer model

        Returns:
            A tuple of two lists:
                - List of topic assignments per document
                - List of sub-topic assignments per document
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
        documents_df["SubTopic"] = -1  # Initialize SubTopic column

        # Update topic sizes
        self._update_topic_sizes(documents_df)
        # Map topics based on frequency
        documents_df = self._map_topics(documents_df)

        # Parse cluster within outlier topics
        while len(documents_df[documents_df.Topic == -1]) > 1000:
            documents_df = self.outlier_cluster_parsing(documents_df)
            # Update topic sizes
            self._update_topic_sizes(documents_df)
            # Map topics based on frequency
            documents_df = self._map_topics(documents_df)

        # Parse sub-clusters within each topic
        documents_df = self.sub_cluster_parsing(documents_df)

        # Save the top 3 most representative documents per topic
        self._save_representative_docs(documents_df, self.umap_embeddings)

        self.probabilities_ = probabilities

        if not self.load_model:
            self.save_results(self.save_path)

        return documents_df.Topic, documents_df.SubTopic

    def _map_topics(self, documents: pd.DataFrame):
        """Map topics based on frequency.

        Arguments:
            documents: A DataFrame containing a 'Topic' column with topic assignments.
        """
        df = pd.DataFrame(self.topic_sizes_.items(), columns=["Old_Topic", "Size"]).sort_values("Size", ascending=False)
        df = df[df.Old_Topic != -1]
        sorted_topics = {**{-1: -1}, **dict(zip(df.Old_Topic, range(len(df))))}

        # Map documents
        documents.Topic = documents.Topic.map(sorted_topics).fillna(documents.Topic).astype(int)

        # Update topic sizes after remapping
        self._update_topic_sizes(documents)

        return documents

    def outlier_cluster_parsing(self, documents: pd.DataFrame):
        """Re-cluster outlier documents to find potential sub-topics."""
        documents_tmp = documents[documents.Topic == -1]
        documents_tmp_umap = self.umap_embeddings[documents_tmp.index.tolist()]
        if len(documents_tmp) > 0:
            # Cluster between the outliers to find potential sub-topics
            hdbscan_outliers = HDBSCAN(
                min_cluster_size=max(2, self.min_topic_size // 5),
                metric="euclidean",
                cluster_selection_method="eom",
                core_dist_n_jobs=-1,
            )
            hdbscan_outliers.fit(documents_tmp_umap, y=None)
            outlier_labels = hdbscan_outliers.labels_
            # add topic labels to documents
            for idx, doc_idx in enumerate(documents_tmp.index.tolist()):
                if outlier_labels[idx] != -1:
                    documents.at[doc_idx, 'Topic'] = len(self.topic_sizes_) + 1 + outlier_labels[idx]

        return documents

    def sub_cluster_parsing(self, documents: pd.DataFrame):
        """Re-cluster documents within each topic to find potential sub-topics."""
        for topic in set(documents.Topic):
            if topic != -1:
                if len(documents[documents.Topic == topic]) > 100:
                    # Get documents and embeddings for this topic
                    topic_docs = documents[documents.Topic == topic]
                    topic_indices = topic_docs.index.tolist()
                    topic_embeddings = self.umap_embeddings[topic_indices]

                    # Re-cluster within the topic
                    hdbscan_sub = HDBSCAN(
                        min_cluster_size=max(2, self.min_topic_size // 5),
                        metric="euclidean",
                        cluster_selection_method="eom",
                        core_dist_n_jobs=-1,
                    )
                    hdbscan_sub.fit(topic_embeddings, y=None)
                    sub_labels = hdbscan_sub.labels_
                    # sort sub_labels to start from 0 except -1 based on frequency
                    sub_label_counts = collections.Counter(sub_labels)
                    sub_label_mapping = {old_label: new_label for new_label, (old_label, _) in enumerate(sub_label_counts.most_common()) if old_label != -1}
                    sub_labels_mapped = [sub_label_mapping[label] if label != -1 else -1 for label in sub_labels]
                    # add sub-topic labels to documents
                    for idx, doc_idx in enumerate(topic_indices):
                        documents.at[doc_idx, 'SubTopic'] = sub_labels_mapped[idx]

        return documents


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

    model = BertTopic_doc()
    result = model.fit_transform((documents['title'] + '\n' + documents['cleaned_text']).tolist(), embeddings=embeddings)
    a = 1