from sentence_transformers import SentenceTransformer, util
import numpy as np
import concurrent.futures
import re
import time


class LogAnalyzer:
    def __init__(self, target_sentences, model_name='paraphrase-MiniLM-L6-v2', threshold=0.8):
        self.model = SentenceTransformer(model_name)
        self.target_sentences = target_sentences
        self.target_embeddings = self._precompute_embeddings(target_sentences)
        self.threshold = threshold

    def _precompute_embeddings(self, sentences):
        return self.model.encode(sentences, convert_to_tensor=True).cpu().numpy()

    def process_log_file(self, log_file_path):
        start_time = time.time()

        with open(log_file_path, 'r') as f:
            lines = f.readlines()

        line_embeddings = self.model.encode(lines, convert_to_tensor=True).cpu().numpy()
        similarities = np.inner(line_embeddings, self.target_embeddings)
        score = np.sum(np.any(similarities > self.threshold, axis=1))

        end_time = time.time()
        print(f"Time taken to process {log_file_path}: {end_time - start_time} seconds")

        return score

    def process_multiple_files(self, file_paths):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            scores = list(executor.map(self.process_log_file, file_paths))
        return scores

    def simple_string_match(self, pattern, log_file_path):
        score = 0
        with open(log_file_path, 'r') as f:
            for line in f:
                if re.search(pattern, line):
                    score += 1
        return score

    def process_files_with_string_match(self, pattern, file_paths):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            scores = list(executor.map(lambda x: self.simple_string_match(pattern, x), file_paths))
        return scores

