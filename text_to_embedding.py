"""
Text Preprocessor Module
Handles HTML cleaning, Korean text tokenization, and preprocessing
"""
import re
import logging
from typing import List, Tuple
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Preprocess Korean text data"""

    def __init__(self, huggingface_model: str = 'LiquidAI/LFM2-350M-Extract', mecab_path='C:/mecab/mecab-ko-dic'):
        """Initialize the text preprocessor"""
        self.mecab = None
        self.mecab_path = mecab_path
        self._initialize_tokenizer()
        self._initialize_huggingface_model(huggingface_model)

    def _initialize_tokenizer(self):
        """Initialize Korean tokenizer (Mecab)"""
        try:
            from konlpy.tag import Mecab
            self.mecab = Mecab(dicpath=self.mecab_path)
            logger.info("Mecab tokenizer initialized successfully")
        except Exception as e:
            logger.warning(f"Mecab not available: {e}. Falling back to simple tokenization")
            self.mecab = None

    def _initialize_huggingface_model(self, huggingface_model: str):
        self.huggingface_model = huggingface_model
        # load model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.huggingface_model)
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # IMPORTANT: For decoder-only models, use left padding
            #self.tokenizer.padding_side = 'left'

            # Use proper torch dtype instead of string
            self.model = AutoModelForCausalLM.from_pretrained(
                self.huggingface_model,
                device_map='auto',
                dtype=torch.bfloat16,
            )

            # Enable gradient checkpointing for memory efficiency (optional)
            # self.model.gradient_checkpointing_enable()

            logger.info(f"Huggingface model loaded successfully on {self.model.device}")
        except Exception as e:
            logger.error(f"Failed to load Huggingface model: {e}")
            self.tokenizer = None
            self.model = None

    def keywordify_text(self, text: str) -> str:
        """
        Extract keywords using Huggingface model (single text)
        For better GPU utilization, use keywordify_text_batch() instead.

        Args:
            text: Text document
        Returns:
            Extracted keywords as string
        """
        if self.tokenizer is None or self.model is None:
            logger.error("Huggingface model is not loaded")
            return ""

        prompt = """
        이 글에서 핵심 주제를 추출해줘
        """
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt + text}],
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to(self.model.device)

        try:
            with torch.inference_mode():
                output = self.model.generate(
                    input_ids,
                    do_sample=True,
                    temperature=0.3,
                    min_p=0.15,
                    repetition_penalty=1.05,
                    max_new_tokens=256,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
            cleansed_output = decoded_output.split('assistant')[-1].replace('"','').replace('\n',' ').replace('(',' ').replace(')',' ').replace('_', ' ').strip()

        except Exception as e:
            logger.error(f"Huggingface extraction failed: {e}")
            cleansed_output = text.replace('"','').replace('\n',' ').replace('(',' ').replace(')',' ').replace('_', ' ').strip()

        return cleansed_output

    def keywordify_text_batch(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """
        Extract keywords using Huggingface model with batch processing for better GPU utilization.

        Args:
            texts: List of text documents
            batch_size: Number of texts to process in parallel (adjust based on GPU memory)
        Returns:
            List of extracted keywords as strings
        """
        if self.tokenizer is None or self.model is None:
            logger.error("Huggingface model is not loaded")
            return [""] * len(texts)

        prompt = """
        이 글에서 중요한 내용을 내포하는 키워드 목록을 추출하세요. 중요하지 않은 광고 문구는 포함하지 마세요.
        """

        all_results = []

        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Keywordifying texts"):
            batch_texts = texts[i:i + batch_size]

            # Prepare chat templates for batch
            batch_messages = [
                [{"role": "user", "content": prompt + text}]
                for text in batch_texts
            ]

            try:
                # Tokenize batch with proper padding
                batch_inputs = []
                for messages in batch_messages:
                    input_ids = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        tokenize=True,
                        enable_thinking=False
                    )
                    batch_inputs.append(input_ids[0])

                # Pad sequences to same length
                from torch.nn.utils.rnn import pad_sequence
                padded_inputs = pad_sequence(
                    batch_inputs,
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id
                ).to(self.model.device)

                # Create attention mask
                #attention_mask = (padded_inputs != self.tokenizer.pad_token_id).long()

                # Generate with batch processing
                with torch.inference_mode():
                    outputs = self.model.generate(
                        padded_inputs,
                        #attention_mask=attention_mask,
                        do_sample=True,
                        temperature=0.3,
                        min_p=0.15,
                        repetition_penalty=1.5,
                        max_new_tokens=128,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                # Decode batch outputs
                for output in outputs:
                    decoded_output = self.tokenizer.decode(output, skip_special_tokens=True)
                    cleansed_output = decoded_output.split('</think>')[-1].replace('"','').replace('\n',' ').replace('(',' ').replace(')',' ').replace('_', ' ').strip()
                    all_results.append(cleansed_output)

            except Exception as e:
                logger.error(f"Batch extraction failed: {e}")
                # Fallback for failed batch
                for text in batch_texts:
                    cleansed = text.replace('"','').replace('\n',' ').replace('(',' ').replace(')',' ').replace('_', ' ').strip()
                    all_results.append(cleansed)

        return all_results

    def clean_html(self, text: str) -> str:
        """
        Remove HTML tags and clean text

        Args:
            text: Raw text with HTML

        Returns:
            Cleaned text without HTML
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Parse HTML
        soup = BeautifulSoup(text, 'lxml')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text

    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text

        Args:
            text: Input text

        Returns:
            Text without URLs
        """
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', text)

    def remove_email(self, text: str) -> str:
        """
        Remove email addresses from text

        Args:
            text: Input text

        Returns:
            Text without emails
        """
        email_pattern = re.compile(r'\S+@\S+')
        return email_pattern.sub('', text)

    def remove_special_chars(self, text: str) -> str:
        """
        Remove special characters but keep Korean, English, numbers

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        # Keep Korean (Hangul), English, numbers, and basic punctuation
        pattern = re.compile(r'[^가-힣a-zA-Z0-9\s.,!?()]+')
        return pattern.sub(' ', text)

    def remove_non_meaningful_chars(self, text: str) -> str:
        """
        Remove non-meaningful characters like isolated consonants/vowels

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        pattern = re.compile(r'[ㄱ-ㅎㅏ-ㅣ]+뉴스')
        pattern.sub(' ', text)

        def replace_match(text, words):
            for word in words:
                text = text.replace(word, ' ')
            return text

        cleansed_text = replace_match(text, [' 무단 ', ' 전재 ', ' 금지 ', ' 학습 ', ' 활용 ', ' 저작 ', ' 주제 ', ' 추출 ', ' 년 ', ' 월 ', ' 일 ', ' 로이터 '])
        return cleansed_text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Korean text using Mecab

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        if not text:
            return []

        if self.mecab:
            try:
                return self.mecab.morphs(text)
            except Exception as e:
                logger.warning(f"Mecab tokenization failed: {e}. Using simple split.")
                return text.split()
        else:
            # Simple whitespace tokenization as fallback
            return text.split()

    def extract_nouns(self, text: str) -> List[str]:
        """
        Extract nouns from Korean text

        Args:
            text: Input text

        Returns:
            List of nouns
        """
        if not text:
            return []

        if self.mecab:
            try:
                return self.mecab.nouns(text)
            except Exception as e:
                logger.warning(f"Noun extraction failed: {e}")
                return []
        else:
            logger.warning("Mecab not available. Cannot extract nouns.")
            return []

    def extract_pos(self, text: str, pos_tags: List[str] = ['NNG', 'NNP', 'SL']) -> List[str]:
        """
        Extract words with specific POS tags

        Args:
            text: Input text
            pos_tags: List of POS tags to extract (default: common nouns and proper nouns)

        Returns:
            List of words matching POS tags
        """
        if not text or not self.mecab:
            return []

        try:
            pos_result = self.mecab.pos(text)
            return [word for word, pos in pos_result if pos in pos_tags]
        except Exception as e:
            logger.warning(f"POS tagging failed: {e}")
            return []

    def preprocess_text(self, text: str, extract_nouns_only: bool = True) -> str:
        """
        Complete preprocessing pipeline

        Args:
            text: Raw text
            extract_nouns_only: If True, return only nouns

        Returns:
            Preprocessed text
        """
        # Clean HTML
        text = self.clean_html(text)

        # Remove URLs and emails
        text = self.remove_urls(text)
        text = self.remove_email(text)

        # Remove special characters
        text = self.remove_special_chars(text)

        # Remove non-meaningful characters
        text = self.remove_non_meaningful_chars(text)

        # Extract nouns if requested
        if extract_nouns_only:
            nouns = self.extract_pos(text)
            return ' '.join(nouns)
        else:
            return text

    def preprocess_dataframe(self, df: pd.DataFrame,
                            text_columns: List[str] = ['title', 'content'],
                            extract_nouns_only: bool = True) -> pd.DataFrame:
        """
        Preprocess text columns in a DataFrame

        Args:
            df: Input DataFrame
            text_columns: Columns to preprocess
            extract_nouns_only: If True, extract only nouns

        Returns:
            DataFrame with preprocessed text
        """
        df = df.copy()

        for column in text_columns:
            if column not in df.columns:
                logger.warning(f"Column '{column}' not found in DataFrame")
                continue

            logger.info(f"Preprocessing column: {column}")
            df[f'{column}_cleaned'] = df[column].apply(
                lambda x: self.preprocess_text(x, extract_nouns_only)
            )

        return df

    def get_vocabulary(self, texts: List[str], min_length: int = 2) -> List[str]:
        """
        Get unique vocabulary from texts

        Args:
            texts: List of texts
            min_length: Minimum word length

        Returns:
            Sorted list of unique words
        """
        vocabulary = set()

        for text in texts:
            words = text.split()
            vocabulary.update(word for word in words if len(word) >= min_length)

        return sorted(vocabulary)
