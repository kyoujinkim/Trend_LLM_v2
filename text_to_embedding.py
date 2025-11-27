# @title Step1: CSV 파일에서 뉴스 데이터 import
import configparser
import gc
import json
import pickle

import pandas as pd
from tqdm import tqdm
from glob import glob
import torch
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import re

class Text2Embedding:
    def __init__(self, config, project_config, stdate=None, enddate=None):
        '''
        Initialize the Text2Embedding class
        :param model_id: model path from huggingface
        :param config: config which contains api keys
        :param project_config: project specific configurations
        :param stdate: start date for data filtering
        :param enddate: end date for data filtering
        '''
        self.config = config
        self.project_config = project_config
        self.hb_token = self.config.get('huggingface','token')
        self.model_id = self.project_config.get('embedding', 'model_id')
        self.batch_size = self.project_config.getint('embedding', 'batch_size')
        self.iterate_size = self.project_config.getint('embedding', 'iterate_size')

        self.model = self._init_embedding_model()
        self.data = self._load_data(stdate, enddate)

    def _init_embedding_model(self):
        '''
        Initialize the embedding model from huggingface
        :return: embedding model
        '''
        login(token=self.hb_token)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(self.model_id).to(device=device)
        return model

    def _load_data(self, stdate, enddate):
        '''
        Load data from CSV files
        :return: concatenated dataframe
        '''
        data_path = self.project_config.get('data', 'data_path')
        fpath = glob(f'{data_path}/*.csv')

        total = []
        for f in tqdm(fpath):
            tmp = pd.read_csv(f, encoding='UTF-8-sig')
            total.append(tmp)

        total_df = pd.concat(total)

        # sort data by 'class' 'date' 'time' 'source' 'kind'
        total_df = total_df.sort_values(by=['class', 'date', 'time', 'source', 'kind'])

        total_df['date_dt'] = pd.to_datetime(total_df['date'], format='%Y%m%d')

        # filter by date range
        if stdate is None:
            stdate = total_df['date_dt'].min()
        if enddate is None:
            enddate = total_df['date_dt'].max()
        total_df = total_df[(total_df['date_dt'] >= pd.to_datetime(stdate)) & (total_df['date_dt'] <= pd.to_datetime(enddate))]

        total_df['cleaned_text'] = total_df['content'].apply(Text2Embedding._extract_text)

        return total_df

    @staticmethod
    def _extract_text(xml_string):
        '''
        Extract text from XML formatted string
        :param xml_string: XML formatted string
        :return: cleaned text
        '''
        clean = re.compile('<.*?>')
        return re.sub(clean, '', xml_string)

    def run(self):
        '''
        Run the text to embedding process
        :return: None
        '''
        import os
        data_path = self.project_config.get('data', 'data_path')

        # Create directories if they don't exist
        os.makedirs(f'{data_path}/embeddings', exist_ok=True)

        # Initialize emb_mapper if it doesn't exist
        emb_mapper_path = f'{data_path}/emb_mapper.json'
        if not os.path.exists(emb_mapper_path):
            with open(emb_mapper_path, 'w') as f:
                json.dump({}, f)

        documents = (self.data['title'] + '\n' + self.data['cleaned_text']).tolist()

        for i in tqdm(range(0, len(documents), self.iterate_size)):
            partial_doc = documents[i:i+self.iterate_size]
            documents_embeddings = self.model.encode(
                partial_doc,
                batch_size=self.batch_size,
                show_progress_bar=True,
            )

            with open(f'{data_path}/embeddings/embeddings{int(i/(self.iterate_size))}.pkl', 'wb') as f:
                pickle.dump(documents_embeddings, f)

            # to ensure embedding map to the correct documents, save with index(class, date, time, source, kind)
            with open(emb_mapper_path, 'r') as f:
                emb_mapper = json.load(f)

            # Get the subset of data for this iteration
            partial_data = self.data.iloc[i:i+self.iterate_size]
            partial_index = partial_data[['class', 'date', 'time', 'source', 'kind']]
            partial_index_joined = partial_index.astype(str).agg('_'.join, axis=1)

            # update emb_mapper
            for j, idx in enumerate(partial_index_joined):
                emb_mapper[idx] = f'embeddings{int(i/(self.iterate_size))}.pkl_{j}'
            # dump
            with open(emb_mapper_path, 'w') as f:
                json.dump(emb_mapper, f)

            gc.collect()

if __name__ == "__main__":
    project_config = configparser.ConfigParser()
    project_config.read('./project_config.ini')
    config = configparser.ConfigParser()
    config.read('D:/config.ini')
    t2e = Text2Embedding(config, project_config)
    t2e.run()