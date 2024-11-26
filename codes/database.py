import os
import numpy as np
import faiss

class VectorDatabase:
    def __init__(self, embedding_dim: int, db_path: str):
        self.embedding_dim = embedding_dim
        self.db_path = db_path
        self.vector_db = None
        
        if not os.path.isfile(self.db_path):
            self.vector_db = self._init_db()
        else:
            self.vector_db = self._load_db()
        
    def _init_db(self):
        index = faiss.IndexFlatL2(self.embedding_dim)
        return index

    def _load_db(self):
        index = faiss.read_index(self.db_path)
        return index
    
    def _save(self):
        faiss.write_index(self.vector_db, self.db_path)
        return True

    def add(self, embedding_matrix: np.ndarray):
        assert self.vector_db is not None
        self.vector_db.add(embedding_matrix)

    def search(self):
        pass