import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer


class RAG_model:

    def __init__(self, rag_datadir="data/rag_data", sentence_transformer_model="all-MiniLM-L6-v2"):
        self.rag_database = json.load(open(f"{rag_datadir}/dev.json.rag", 'r', encoding='utf-8'))

        # Load the pretrained Sentence Transformer model
        self.model = SentenceTransformer(sentence_transformer_model)

        # Check if GPU is available and move the model to GPU if possible
        if torch.cuda.is_available():
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)  # Move model to GPU
            print("[RAG INFO]: Using GPU for inference.")
        else:
            self.device = torch.device("cpu")
            print("[RAG INFO]: Using CPU for inference.")

        # The sentences to encode
        sentences = ["Label: " +item['ground_truth'] + "\n" + item['input'] for item in self.rag_database]

        # Calculate embeddings by calling model.encode()
        self.db_embeddings = self.model.encode(sentences, convert_to_tensor=True, show_progress_bar=True)
        # Move embeddings to the GPU (if available)
        self.db_embeddings = self.db_embeddings.to(self.device)
        print(f"[RAG INFO]: Embeddings shape: {self.db_embeddings.shape}")

    def query(self, query, k=1):
        print(f"Query requested: {query}")
        # Calculate the query embedding
        query_embedding = self.model.encode([query], convert_to_tensor=True)
        query_embedding = query_embedding.to(self.device)  # Move query to GPU if available

        # Calculate cosine similarities between the query and all database embeddings
        similarities = torch.nn.functional.cosine_similarity(query_embedding, self.db_embeddings)

        # Get the top k indices of the highest similarity values
        top_k_indices = similarities.cpu().numpy().argsort()[-k:][::-1]

        # Retrieve the corresponding top k texts
        top_k_texts = [self.rag_database[idx] for idx in top_k_indices]

        return_format = "\n<query_response>\nThe most similar history judge cases are as follows: \n{}\n</query_response>\n\n<judge>\n"

        return return_format.format(str(top_k_texts))
