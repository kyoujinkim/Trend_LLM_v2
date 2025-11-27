# pip install umap-learn
import collections
import json
import os
import pickle
from glob import glob
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from sklearn.cluster import HDBSCAN as SK_HDBSCAN
from sklearn.preprocessing import normalize

class BertTopic_morph():

    def __init__(self,
        language: str = "multilingual",
        top_n_words: int = 10,
        n_gram_range: Tuple[int, int] = (1, 1),
        min_topic_size: int = 10,
        nr_topics: Union[int, str] = None,
        low_memory: bool = False,
        calculate_probabilities: bool = False,
        seed_topic_list: List[List[str]] = None,
        zeroshot_topic_list: List[str] = None,
        zeroshot_min_similarity: float = 0.7,
        embedding_model=None,
        verbose: bool = False,
    ):
        self.language = language
        self.top_n_words = top_n_words
        self.n_gram_range = n_gram_range
        self.min_topic_size = min_topic_size
        self.nr_topics = nr_topics
        self.low_memory = low_memory
        self.calculate_probabilities = calculate_probabilities
        self.seed_topic_list = seed_topic_list
        self.zeroshot_topic_list = zeroshot_topic_list
        self.zeroshot_min_similarity = zeroshot_min_similarity
        self.embedding_model = embedding_model
        self.verbose = verbose

        self.vectorizer_model = CountVectorizer(ngram_range=self.n_gram_range)
        self.umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            low_memory=self.low_memory,
        )

        self.hdbscan_model = SK_HDBSCAN(
            min_cluster_size=self.min_topic_size, metric="euclidean", cluster_selection_method="eom", n_jobs=-1
        )

        # Public attributes
        self.topics_ = None
        self.probabilities_ = None
        self.topic_sizes_ = None
        self.topic_mapper_ = None
        self.topic_representations_ = None
        self.topic_embeddings_ = None
        self._topic_id_to_zeroshot_topic_idx = {}
        self.custom_labels_ = None
        self.c_tf_idf_ = None
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

    def _c_tf_idf(self, documents_per_topic: pd.DataFrame, fit: bool = True):
        """Calculate c-TF-IDF scores for each topic.

        Arguments:
            documents_per_topic: DataFrame with 'Topic' and 'Document' columns
            fit: Whether to fit the vectorizer

        Returns:
            c_tf_idf: c-TF-IDF matrix
        """
        documents = documents_per_topic.Document.values
        if fit:
            self.vectorizer_model.fit(documents)

        c_tf_idf = self.vectorizer_model.transform(documents)

        # Calculate IDF
        n_docs = len(documents)
        df = np.asarray(c_tf_idf.sum(axis=0)).flatten()
        idf = np.log(n_docs / (df + 1))

        # Calculate c-TF-IDF
        c_tf_idf = c_tf_idf.multiply(idf)
        c_tf_idf = normalize(c_tf_idf, axis=1, norm='l1')

        return c_tf_idf

    def _extract_words_per_topic(self, c_tf_idf: np.ndarray, labels: List[int] = None):
        """Extract words per topic.

        Arguments:
            c_tf_idf: c-TF-IDF matrix
            labels: List of topic labels

        Returns:
            topic_representations: Dictionary of topic representations
        """
        if labels is None:
            labels = sorted(list(self.topic_sizes_.keys()))

        # Get feature names from vectorizer
        feature_names = self.vectorizer_model.get_feature_names_out()

        topic_representations = {}
        for index, topic in enumerate(labels):
            if topic != -1:
                # Get top words for this topic
                words_idx = np.argsort(c_tf_idf[index].toarray()[0])[::-1][:self.top_n_words]
                words = [(feature_names[idx], c_tf_idf[index, idx]) for idx in words_idx]
                topic_representations[topic] = words

        return topic_representations

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

        # UMAP dimensionality reduction
        umap_embeddings = self.umap_model.fit_transform(embeddings, y=None)

        # Clustering with HDBSCAN
        self.hdbscan_model.fit(umap_embeddings, y=None)
        labels = self.hdbscan_model.labels_
        documents_df["Topic"] = labels

        # Update topic sizes
        self._update_topic_sizes(documents_df)

        probabilities = self.hdbscan_model.probabilities_

        # Map topics based on frequency
        df = pd.DataFrame(self.topic_sizes_.items(), columns=["Old_Topic", "Size"]).sort_values("Size", ascending=False)
        df = df[df.Old_Topic != -1]
        sorted_topics = {**{-1: -1}, **dict(zip(df.Old_Topic, range(len(df))))}

        # Map documents
        documents_df.Topic = documents_df.Topic.map(sorted_topics).fillna(documents_df.Topic).astype(int)

        # Update topic sizes after remapping
        self._update_topic_sizes(documents_df)

        # Create documents per topic
        documents_per_topic = documents_df.groupby(["Topic"], as_index=False).agg({"Document": " ".join})

        # Extract topic representations using c-TF-IDF
        self.c_tf_idf_ = self._c_tf_idf(documents_per_topic)
        self.topic_representations_ = self._extract_words_per_topic(self.c_tf_idf_)

        # Save the top 3 most representative documents per topic
        self._save_representative_docs(documents_df, umap_embeddings)

        predictions = documents_df.Topic.to_list()
        self.probabilities_ = probabilities

        return predictions

    def save_results(self, output_path: str, documents_df: pd.DataFrame = None):
        """Save clustering results to files.

        Arguments:
            output_path: Directory to save results
            documents_df: Optional dataframe with documents and topic assignments
        """
        os.makedirs(output_path, exist_ok=True)

        # Save topic representations
        topic_info = []
        for topic_id, words in self.topic_representations_.items():
            topic_info.append({
                'topic_id': topic_id,
                'size': self.topic_sizes_[topic_id],
                'top_words': [w[0] for w in words],
                'word_scores': [float(w[1]) for w in words],
                'label': self.topic_labels_[topic_id]
            })

        with open(f'{output_path}/topic_info.json', 'w', encoding='utf-8') as f:
            json.dump(topic_info, f, ensure_ascii=False, indent=2)

        # Save representative documents
        with open(f'{output_path}/representative_docs.json', 'w', encoding='utf-8') as f:
            json.dump(self.representative_docs_, f, ensure_ascii=False, indent=2)

        # Save topic assignments if documents provided
        if documents_df is not None:
            documents_df.to_csv(f'{output_path}/topic_assignments.csv', index=False, encoding='utf-8-sig')

        # Save model components
        with open(f'{output_path}/model_components.pkl', 'wb') as f:
            pickle.dump({
                'umap_model': self.umap_model,
                'hdbscan_model': self.hdbscan_model,
                'vectorizer_model': self.vectorizer_model,
                'c_tf_idf': self.c_tf_idf_
            }, f)

        if self.verbose:
            print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # Import Embeddings and Documents
    embed_list = glob('./data/embeddings/*.pkl')
    embeddings = np.concatenate([pd.read_pickle(f) for f in embed_list], axis=0)
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