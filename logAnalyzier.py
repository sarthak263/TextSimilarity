import os
import torch
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
from concurrent.futures import ProcessPoolExecutor
from CheckPoints import CheckPointsEnum

# Constants
MODEL_NAMES = ["multi-qa-mpnet-base-dot-v1", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-dot-v1", 'paraphrase-MiniLM-L6-v2']
EMBEDDING_DIR = "storedEmbeddings"
EMBEDDING_FILE_PATH = os.path.join(EMBEDDING_DIR, "targetEmbeddings.pkl")
THRESHOLD = 0.8

class LogAnalyzer:

    def __init__(self,model_name=MODEL_NAMES[1]):
        self.model = SentenceTransformer(model_name)
        self.target_sentences = [checkpoint.value for checkpoint in CheckPointsEnum]
        self.target_embeddings = []
        self._initialize_embeddings()

    def _initialize_embeddings(self) -> None:
        if not self._load_embeddings():
            for each_CP_sentences in self.target_sentences:
                self.target_embeddings.append(self._precompute_embeddings(each_CP_sentences))
            self._save_embeddings()

    def _precompute_embeddings(self, sentences) -> torch.Tensor:
        return self.model.encode(sentences, convert_to_tensor=True)

    def _load_embeddings(self) -> bool:
        if Path(EMBEDDING_FILE_PATH).exists():
            with open(EMBEDDING_FILE_PATH, "rb") as fin:
                self.target_embeddings = pickle.load(fin)
                return True
        return False

    def _save_embeddings(self) -> None:
        if not os.path.exists(EMBEDDING_DIR):
            os.makedirs(EMBEDDING_DIR)
        with open(EMBEDDING_FILE_PATH, 'wb') as fout:
            pickle.dump(self.target_embeddings, fout,protocol=pickle.HIGHEST_PROTOCOL)

    def process_log_file(self, log_file_path: str) -> dict['str',dict]:
        res = {}

        lines = self._get_sentences(log_file_path)
        line_embeddings = self.model.encode(lines, convert_to_tensor=True)

        for checkpoint_name, (target, embedding) in zip(CheckPointsEnum.__members__, zip(self.target_sentences, self.target_embeddings)):
            score  = 0
            for line, target_sentence in zip(target, embedding):
                hits = util.semantic_search(target_sentence, line_embeddings, top_k=1)[0][0]
                sc = round(hits['score'],2)
                if sc >= THRESHOLD:
                    #print(line, hits['score'])
                    score += 1
            res[checkpoint_name] = score

        return {log_file_path: res}

    def process_multiple_files(self, file_paths: list) -> list:
        with ProcessPoolExecutor(max_workers=3) as executor:
            return list(executor.map(self.process_log_file, file_paths))

    @staticmethod
    def _get_sentences(file_path: str) -> list:
        extracted_sentences = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if "=>" in line:
                    after_symbol = line.split("=>", 1)[1].strip()
                    extracted_sentences.extend(sent_tokenize(after_symbol))
        return extracted_sentences
