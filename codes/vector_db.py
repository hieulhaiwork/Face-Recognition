import os
import numpy as np
from typing import List, Tuple

import faiss
import json

from utils import configs

class VectorDatabase:
    def __init__(self, embedding_dim: int=768, db_path: str = None, quantized: bool=True):
        self.embedding_dim = embedding_dim
        self.quantized = quantized
        self.vector_db = None
        self.names_path = configs.get("names").get("path")

        if not os.path.isfile(self.names_path):
            self.names_dict = self._init_names_file(create=True)
        else:
            self.names_dict = self._init_names_file(create=False)
        
        self.names_list = self.names_dict.get("names")
        assert type(self.names_list) != 'list'

        if db_path is None:
            self.db_path = configs.get("database", {}).get("path", "")
            assert self.db_path != ""
        else:
            self.db_path = db_path
        
        if not os.path.isfile(self.db_path):
            self.vector_db = self._init_db()
        else:
            self.vector_db = self._load_db()
        
    def _init_db(self):
        """
        Initialize vector database with  Product Quantization (PQ) in Faiss to reduce the storage and increase retrieval speed.
        """
        if self.quantized:
            index = faiss.IndexPQ(self.embedding_dim, 8, 8)
        else:
            index = faiss.IndexFlatL2(self.embedding_dim)
        
        return index

    def _load_db(self):
        index = faiss.read_index(self.db_path)
        return index
    
    def _init_names_file(self, create: bool = True):
        if create:
            with open(self.names_path, 'w') as names_file:
                json.dump({
                    "names": []
                }, names_file, indent=4)

        with open(self.names_path, 'r') as n_file:
            names = json.load(n_file)
    
        return names
    
    def _save(self):
        """
        Save database and name list to retrieval later:
            - Vector database: to path in configs.json file: database/face_db.faiss
            - Names: to path in configs.json file: database/names.json
        """
        assert self.vector_db is not None and self.names_list is not None, "There's nothing to save."
        # Save database
        faiss.write_index(self.vector_db, self.db_path)

        # Save names list
        self.names_dict["names"] = self.names_list
        with open(self.names_path, 'w') as json_file:
            json.dump(self.names_dict, json_file, indent=4)

        return True

    def _add(self, embedding_matrix: np.ndarray):
        assert self.vector_db is not None
        self.vector_db.add(embedding_matrix.reshape(1, -1))
        return True

    def _search(self, embedding_matrix: np.ndarray, k: int = 1):
        
        vector_db = self._load_db()
        names = self._init_names_file(create=False).get("names")

        distances, indices = vector_db.search(embedding_matrix, k)
        name = names[indices[0][0]]

        return distances, indices[0][0], name
    
    def new(self, image_list: List[np.ndarray]):
        assert len(image_list) > 0
        for image in image_list:
            self._add(image)
            name = input("Input name of this person: ")
            self.names_list.append(name)
        self._save()
        return True
    
    def get_name(self, embedding: np.ndarray, k: int = 1):
        distance, index, name = self._search(embedding.reshape(1, -1), k)
        return distance, index, name
        
