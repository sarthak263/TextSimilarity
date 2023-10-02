import os
import torch
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
import concurrent.futures
from CheckPoints import CheckPointsEnum

# Constants
MODEL_NAMES = ["multi-qa-mpnet-base-dot-v1", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-dot-v1", 'paraphrase-MiniLM-L6-v2']
EMBEDDING_DIR = "storedEmbeddings"
EMBEDDING_FILE_PATH = os.path.join(EMBEDDING_DIR, "targetEmbeddings.pkl")
THRESHOLD = 0.8

class LogAnalyzer:
    """Class responsible for analyzing logs against predefined checkpoints."""

    def __init__(self, model_name=MODEL_NAMES[1]):
        """Initialize the model and embeddings."""
        self.model = SentenceTransformer(model_name)
        self.target_sentences = [checkpoint.value for checkpoint in CheckPointsEnum]
        self.target_embeddings = []
        self._initialize_embeddings()

    def _initialize_embeddings(self) -> None:
        """Load embeddings from file or compute if they don't exist."""
        if not self._load_embeddings():
            self.target_embeddings = [self._precompute_embeddings(sentences) for sentences in self.target_sentences]
            self._save_embeddings()

    def _precompute_embeddings(self, sentences) -> torch.Tensor:
        """Compute embeddings for the given sentences."""
        return self.model.encode(sentences, convert_to_tensor=True)

    def _load_embeddings(self) -> bool:
        """Load embeddings from file if it exists."""
        if Path(EMBEDDING_FILE_PATH).exists():
            with open(EMBEDDING_FILE_PATH, "rb") as fin:
                self.target_embeddings = pickle.load(fin)
                return True
        return False

    def _save_embeddings(self) -> None:
        """Save embeddings to a file."""
        if not os.path.exists(EMBEDDING_DIR):
            os.makedirs(EMBEDDING_DIR)
        with open(EMBEDDING_FILE_PATH, 'wb') as fout:
            pickle.dump(self.target_embeddings, fout, protocol=pickle.HIGHEST_PROTOCOL)

    def process_log_file(self, log_file_path: str) -> dict[str, dict]:
        """Process a single log file and return results for checkpoints."""
        res = {}

        lines = self._get_sentences(log_file_path)
        line_embeddings = self.model.encode(lines, convert_to_tensor=True)

        # Loop over each checkpoint and its corresponding target and embedding
        for checkpoint_name, (target, embedding) in zip(CheckPointsEnum.__members__, zip(self.target_sentences, self.target_embeddings)):
            score = sum(
                1 for line, target_sentence in zip(target, embedding)
                if util.semantic_search(target_sentence, line_embeddings, top_k=1)[0][0]['score'] >= THRESHOLD
            )
            res[checkpoint_name] = score

        return {log_file_path: res}

    def process_multiple_files(self, file_paths: list) -> list:
        """Process multiple log files concurrently."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(executor.map(self.process_log_file, file_paths))

    @staticmethod
    def _get_sentences(file_path: str) -> list:
        """Extract and return list of sentences from a log file."""
        extracted_sentences = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if "=>" in line:
                    after_symbol = line.split("=>", 1)[1].strip()
                    extracted_sentences.extend(sent_tokenize(after_symbol))
        return extracted_sentences
