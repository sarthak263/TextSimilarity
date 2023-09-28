import torch
from sentence_transformers import SentenceTransformer, util
import concurrent.futures
import nltk

#nltk.download('punkt')
from nltk.tokenize import sent_tokenize

modelName = ["multi-qa-mpnet-base-dot-v1","all-mpnet-base-v2","multi-qa-MiniLM-L6-dot-v1",'paraphrase-MiniLM-L6-v2']

class LogAnalyzer:
    def __init__(self, target_sentences, model_name=modelName[1], threshold=0.8):
        self.model = SentenceTransformer(model_name)
        self.target_sentences = target_sentences
        self.target_embeddings = self._precompute_embeddings(target_sentences)
        self.threshold = threshold

    def _precompute_embeddings(self, sentences):
        return self.model.encode(sentences,convert_to_tensor=True).numpy()

    def process_log_file(self, log_file_path):
        score = 0

        lines  = self.getSentences(log_file_path)
        line_embeddings = self.model.encode(lines, convert_to_tensor=True).cpu().numpy()

        for line, target in zip(self.target_sentences,self.target_embeddings):
            hits = util.semantic_search(target,line_embeddings,top_k=1)[0][0]
            sc = round(hits['score'],2)
            if sc  >=0.85:
                print(line,sc)
                score+=1

        #hits = util.semantic_search(self.target_embeddings,line_embeddings,top_k=1)[0][0]
        #score = np.sum(np.any(hits['score'] > self.threshold, axis=1))
        return log_file_path,score

    def process_multiple_files(self, file_paths):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            res = list(executor.map(self.process_log_file, file_paths))
        return res

    def getSentences(self, file_path) -> list:
        extracted_sentences = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Check if "=>" is in the line
                if "=>" in line:
                    # Find the part of the line after "=>"
                    after_symbol = line.split("=>", 1)[1].strip()

                    # Tokenize the sentences and add them to the list
                    extracted_sentences.extend(sent_tokenize(after_symbol))
        return extracted_sentences
