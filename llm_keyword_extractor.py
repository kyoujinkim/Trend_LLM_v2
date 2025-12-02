"""
LLM Keyword Extractor
Uses LLM to extract representative keywords and phrases from topic clusters
"""

import json
import os
from typing import List, Dict
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from tqdm import tqdm
from pydantic import BaseModel


class KeyPhrases(BaseModel):
    LisfOfKeyphrase: List[str]


class LLMKeywordExtractor:
    """Extract keywords from clusters using LLM"""

    def __init__(self, config, model_class="huggingface", model_id="google/gemma-2-2b-it", max_keywords=10):
        """
        Initialize LLM keyword extractor

        Args:
            config: Configuration with API keys
            model_class: Class of the model (default: 'huggingface', 'openai')
            model_id: model ID for LLM(default: 'google/gemma-2-2b-it')
            max_keywords: Maximum number of keywords to extract per cluster
        """
        self.config = config
        self.model_class = model_class
        self.model_id = model_id
        self.max_keywords = max_keywords

        if model_class == "huggingface":
            self.token = self.config.get('huggingface', 'token')
        elif model_class == 'openai':
            self.token = self.config.get('openai', 'API_KEY')
        else:
            raise ValueError(f"Unsupported model class: {model_class}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

    def _init_model(self):
        """Initialize the LLM model"""
        if self.model is not None:
            return

        if self.model_class == "huggingface":
            login(token=self.token)
            print(f"Loading LLM model: {self.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            print(f"Model loaded on {self.device}")
        elif self.model_class == 'openai':
            self.model = OpenAI()

    def extract_keywords_from_structured_texts(self, texts: List[str], topic_id: int = None) -> List[str]:
        """
        Extract keywords from a list of representative texts

        Args:
            texts: List of representative documents from a cluster
            topic_id: Optional topic ID for tracking

        Returns:
            List of extracted keywords/phrases
        """
        self._init_model()

        # Combine texts (limit to avoid token overflow)
        combined_text = "\n".join(texts[:5])  # Use top 5 representative docs

        # Truncate if too long
        max_chars = 2000
        if len(combined_text) > max_chars:
            combined_text = combined_text[:max_chars]

        # Create prompt for Korean text
        prompt = f"""다음 텍스트들에서 공통적으로 나타나는 주요 키워드와 핵심 주제를 {self.max_keywords}개 추출해주세요.
각 키워드는 2-4 단어로 구성된 명사구 형태로 작성해주세요.

텍스트:
{combined_text}

주요 키워드 ({self.max_keywords}개):"""

        response = self.model.responses.parse(
            model=self.model_id,
            input=[
                {"role": "system", "content": "Extract the event information."},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            text_format=KeyPhrases,
        )

        parsed = response.output_parsed

        # Extract keywords from response
        keywords = parsed.LisfOfKeyphrase[:self.max_keywords]

        return keywords

    def extract_keywords_from_texts(self, texts: List[str], topic_id: int = None) -> List[str]:
        """
        Extract keywords from a list of representative texts

        Args:
            texts: List of representative documents from a cluster
            topic_id: Optional topic ID for tracking

        Returns:
            List of extracted keywords/phrases
        """
        self._init_model()

        # Combine texts (limit to avoid token overflow)
        combined_text = "\n".join(texts[:5])  # Use top 5 representative docs

        # Truncate if too long
        max_chars = 2000
        if len(combined_text) > max_chars:
            combined_text = combined_text[:max_chars]

        # Create prompt for Korean text
        prompt = f"""다음 텍스트들에서 공통적으로 나타나는 주요 키워드와 핵심 주제를 {self.max_keywords}개 추출해주세요.
각 키워드는 2-4 단어로 구성된 명사구 형태로 작성하고, 쉼표로 구분해주세요.

텍스트:
{combined_text}

주요 키워드 ({self.max_keywords}개):"""

        # Generate keywords
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract keywords from response
        # Look for the part after the prompt
        if "주요 키워드" in response:
            keywords_part = response.split("주요 키워드")[-1]
            # Remove the number indication
            keywords_part = keywords_part.split(":", 1)[-1].strip()
            # Split by comma and clean
            keywords = [kw.strip() for kw in keywords_part.split(",")]
            keywords = [kw for kw in keywords if kw and len(kw) > 1][:self.max_keywords]
        else:
            keywords = []

        return keywords

    def extract_keywords_for_all_topics(
        self,
        representative_docs: Dict[int, Dict],
        save_path: str = None
    ) -> Dict[int, List[str]]:
        """
        Extract keywords for all topics

        Args:
            representative_docs: Dictionary mapping topic IDs to their representative documents
                                Format: {topic_id: {'text': [doc1, doc2, ...], 'subdocs': {...}}}
            save_path: Path to save the extracted keywords

        Returns:
            Dictionary mapping topic IDs to their keywords
        """
        topic_keywords = {}

        print(f"Extracting keywords for {len(representative_docs)} topics...")
        for topic_id, docs_info in tqdm(representative_docs.items(), desc="Extracting keywords"):
            if topic_id == -1:
                continue

            texts = docs_info.get('text', [])
            if texts:
                if self.model_class == "huggingface":
                    keywords = self.extract_keywords_from_texts(texts, topic_id)
                elif self.model_class == 'openai':
                    keywords = self.extract_keywords_from_structured_texts(texts, topic_id)
                else:
                    raise ValueError(f"Unsupported model class: {self.model_class}")
                topic_keywords[topic_id] = keywords
            else:
                topic_keywords[topic_id] = []

        # Save results if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(topic_keywords, f, ensure_ascii=False, indent=2)
            print(f"Keywords saved to {save_path}")

        return topic_keywords

    def extract_keywords_for_subtopics(
        self,
        representative_docs: Dict[int, Dict],
        save_path: str = None
    ) -> Dict[int, Dict[int, List[str]]]:
        """
        Extract keywords for all subtopics within topics

        Args:
            representative_docs: Dictionary with topic representative documents including subdocs
            save_path: Path to save the extracted keywords

        Returns:
            Nested dictionary mapping topic IDs to subtopic IDs to their keywords
        """
        subtopic_keywords = {}

        print(f"Extracting keywords for subtopics...")
        for topic_id, docs_info in tqdm(representative_docs.items(), desc="Processing topics"):
            if topic_id == -1:
                continue

            subdocs = docs_info.get('subdocs', {})
            if subdocs:
                subtopic_keywords[topic_id] = {}
                for subtopic_id, texts in subdocs.items():
                    if texts:
                        keywords = self.extract_keywords_from_texts(texts, subtopic_id)
                        subtopic_keywords[topic_id][subtopic_id] = keywords

        # Save results if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(subtopic_keywords, f, ensure_ascii=False, indent=2)
            print(f"Subtopic keywords saved to {save_path}")

        return subtopic_keywords

    def cleanup(self):
        """Free up memory"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None


if __name__ == "__main__":
    # Example usage
    import configparser

    config = configparser.ConfigParser()
    config.read('D:/config.ini')

    extractor = LLMKeywordExtractor(config)

    # Example representative docs
    sample_docs = {
        0: {
            'text': [
                "삼성전자가 새로운 반도체 공장을 건설한다고 발표했다.",
                "반도체 산업의 성장세가 지속되고 있다.",
                "메모리 반도체 수요가 급증하고 있다."
            ],
            'subdocs': {}
        }
    }

    keywords = extractor.extract_keywords_for_all_topics(sample_docs)
    print(keywords)