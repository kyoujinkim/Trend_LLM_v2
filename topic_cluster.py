# pip install umap-learn
import collections
from glob import glob
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from sklearn.cluster import HDBSCAN as SK_HDBSCAN

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
        documents = pd.DataFrame({"Document": documents, "ID": doc_ids, "Topic": None})

        umap_embeddings = self.umap_model.fit_transform(embeddings, y=None)

        self.hdbscan_model.fit(umap_embeddings, y=None)

        labels = self.hdbscan_model.labels_

        documents["Topic"] = labels

        probabilities = self.hdbscan_model.probabilities_

        # Map topics based on frequency
        df = pd.DataFrame(self.topic_sizes_.items(), columns=["Old_Topic", "Size"]).sort_values("Size", ascending=False)
        df = df[df.Old_Topic != -1]
        sorted_topics = {**{-1: -1}, **dict(zip(df.Old_Topic, range(len(df))))}

        # Map documents
        documents.Topic = documents.Topic.map(sorted_topics).fillna(documents.Topic).astype(int)

        documents_per_topic = documents.groupby(["Topic"], as_index=False).agg({"Document": " ".join})

        # Save the top 3 most representative documents per topic
        #self._save_representative_docs(documents)

        predictions = documents.Topic.to_list()

        return predictions

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