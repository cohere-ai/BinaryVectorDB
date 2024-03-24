import faiss
import numpy as np
from rocksdict import Rdict
import os
import time 
import logging 
import cohere
import json
import tqdm
from typing import List


logger = logging.getLogger(__name__)

class BinaryVectorDB:
    def __init__(self, folder, model = "embed-multilingual-v3.0", index_type=faiss.IndexBinaryFlat, index_args=[1024], rdict_options=None):
        if 'COHERE_API_KEY' not in os.environ:
            raise Exception("Please set the COHERE_API_KEY environment variable to your Cohere API key.")
        
        self.co = cohere.Client(os.environ['COHERE_API_KEY']) 

        config_path = os.path.join(folder, "config.json")
        if not os.path.exists(config_path):
            if os.path.exists(folder) and len(os.listdir(folder)) > 0:
                raise Exception(f"Folder {folder} contains files, but no config.json. If you want to create a new CohereBinaryVectorDB, the folder must be empty. If you want to load an existing CohereBinaryVectorDB, the folder must contain a config.json file that defines the model.")
            
            os.makedirs(folder, exist_ok=True)
            with open(config_path, "w") as fOut:
                config = {'version': '1.0', 'model': model}
                json.dump(config, fOut)
        
        with open(config_path, "r") as fIn:
            self.config = json.load(fIn)

        os.makedirs(folder, exist_ok=True)
        self.folder = folder

        faiss_index_path = os.path.join(folder, "index.bin")
        if not os.path.exists(faiss_index_path):
            self.index = faiss.IndexBinaryIDMap2(index_type(*index_args))
        else:
            self.index = faiss.read_index_binary(faiss_index_path)

        self.doc_db = Rdict(os.path.join(folder, "docs"), rdict_options)
        

    def add_documents(self, doc_ids, docs, docs2text=lambda x: x, batch_size = 960, save=True):
        """
        Add documents to the index.
        
        Args:
            doc_ids: List of unique ids for each document. Must be either a list of ints or a numpy array of ints
            docs: List of documents. Each document can be any object. The docs2text function will be called on each document to convert it to a string.
            docs2text: Function that converts a document to a string. The default is the identity function.
            batch_size: Number of documents to add in each batch. Default is 960.
            save: Save index after adding the docs. Default is True.
        """
        if len(doc_ids) != len(docs):
            raise ValueError(f"ids and docs must have the same length. Got {len(doc_ids)} doc_ids and {len(docs)} docs.")

        if isinstance(doc_ids, np.ndarray):
            doc_ids = doc_ids.tolist()

        #Extract the texts from the docs
        texts = []
        for doc in docs:
            text = docs2text(doc)
            if not isinstance(text, str):
                raise ValueError(f"docs2text(doc) should return a string, but returned {type(text)}. Change the function call do: db.add_documents(your_docs, lambda x: ...) with a lambda function that transforms your doc to a string.")
            texts.append(text)

        #Delete any existing documents with the same id
        existing_ids = []
        for idx in doc_ids:
            if not isinstance(idx, int):
                raise ValueError(f"doc_id {idx} is not an int.")
            if idx in self.doc_db:
                existing_ids.append(idx)
        
        for idx in existing_ids:
            self.remove_doc(idx, save=False)
        
        #Encode and add the new documents
        with tqdm.tqdm(total=len(texts), desc="Indexing docs") as pBar:
            for start_idx in range(0, len(texts), batch_size):
                batch_text = texts[start_idx:start_idx+batch_size]
                batch_docs = docs[start_idx:start_idx+batch_size]
                batch_ids = doc_ids[start_idx:start_idx+batch_size]
                emb = self.co.embed(texts=batch_text, model=self.config['model'], input_type="search_document", embedding_types=["int8", "ubinary"]).embeddings

                self._add_batch(batch_ids, batch_docs, emb.ubinary, emb.int8)
                pBar.update(len(batch_text))

        if save:
            self.save()

    def _add_batch(self, doc_ids, docs, emb_ubinary, emb_int8):
        if not isinstance(emb_ubinary, np.ndarray):
            emb_ubinary = np.asarray(emb_ubinary, dtype=np.uint8)

        if not isinstance(emb_int8, np.ndarray):
            emb_int8 = np.asarray(emb_int8, dtype=np.int8)

        if not isinstance(doc_ids, np.ndarray):
            doc_ids = np.asarray(doc_ids, dtype=np.int32)
        
        if doc_ids.dtype != np.int32 and doc_ids.dtype != np.int64:
            raise ValueError(f"doc_ids must be a numpy array of np.int32. But got {doc_ids.dtype}")

        assert len(doc_ids) == len(docs)
        assert len(docs) == len(emb_ubinary)
        assert len(emb_ubinary) == len(emb_int8)

        start_time = time.time()
        self.index.add_with_ids(emb_ubinary, doc_ids)
        logger.info(f"Adding binary embeddings to index took {(time.time()-start_time):.3f} s")

        for idx in range(len(docs)):
            self._add_doc(doc_ids[idx].item(), docs[idx], emb_int8[idx]) 
    

    def _add_doc(self, doc_id, doc, emb_int8):
        if not isinstance(emb_int8, np.ndarray):
            emb_int8 = np.asarray(emb_int8, dtype=np.int8)

        self.doc_db[doc_id] = {'doc': doc, 'emb_int8':  emb_int8} 

    def remove_doc(self, doc_id: int, save=True):
        if doc_id not in self.doc_db:
            raise ValueError(f"Document with id {doc_id} not found.")
        
        self.index.remove_ids(np.asarray([doc_id]))
        del self.doc_db[doc_id]

        if save:
            self.save()

    def save(self):
        faiss.write_index_binary(self.index, os.path.join(self.folder, "index.bin"))

    def search(self, query, k=10, binary_oversample=10, int8_oversample=3):
        if self.index.ntotal == 0:
            raise Exception("No documents indexed. Please add documents before searching.")
        
        query_emb = self.co.embed(texts=[query], 
                                   model=self.config['model'],
                                   input_type="search_query", 
                                   embedding_types=["float", "ubinary"]).embeddings

        return self._search_emb(query_emb.float, query_emb.ubinary, k=k, binary_oversample=10, int8_oversample=3)

    def _search_emb(self, query_emb_float, query_emb_ubinary, k=10, binary_oversample=10, int8_oversample=3):
        binary_k = min(k*binary_oversample, self.index.ntotal)
        int8_rescore = k*int8_oversample

        query_emb_ubinary = np.asarray(query_emb_ubinary, dtype=np.uint8)
        
        # Phase I: Search on the index with a binary
        start_time = time.time()
        hits_scores, hits_doc_ids = self.index.search(query_emb_ubinary, k=binary_k)

        #Get the results in a list of hits
        hits = [{'doc_id': doc_id.item(), 'score_hamming': score_bin} for doc_id, score_bin in zip(hits_doc_ids[0], hits_scores[0])]

        logger.info(f"Search with hamming distance took {(time.time()-start_time)*1000:.2f} ms")

        # Phase II: Do a re-scoring with the float query embedding
        start_time = time.time()
        doc_emb_binary = np.asarray([self.index.reconstruct(hit['doc_id']) for hit in hits])
        doc_emb_unpacked = np.unpackbits(doc_emb_binary, axis=-1).astype("int")
        doc_emb_unpacked = 2*doc_emb_unpacked-1

        scores2 = (query_emb_float[0] @ doc_emb_unpacked.T)
        for idx in range(len(scores2)):
            hits[idx]['score_binary'] = scores2[idx]

        #Sort by largest score2
        hits.sort(key=lambda x: x['score_binary'], reverse=True)
        hits = hits[0:int8_rescore]

        logger.info(f"phase2 (float, binary) rescoring took {(time.time()-start_time)*1000:.2f} ms")
        
        # Phase III: Do a re-scoring with the int8 doc embedding
        start_time = time.time()
        doc_emb_int8 = []
        for hit in hits:
            doc = self.doc_db[hit['doc_id']]
            doc_emb_int8.append(doc['emb_int8'])
            hit['doc'] = doc['doc']
        doc_emb_int8 = np.vstack(doc_emb_int8)
        scores3 = (query_emb_float[0] @ doc_emb_int8.T) / np.linalg.norm(doc_emb_int8, axis=1)

        for idx in range(len(scores3)):
            hits[idx]['score_cossim'] = scores3[idx]

        hits.sort(key=lambda x: x['score_cossim'], reverse=True)
        hits = hits[0:k]

        logger.info(f"phase3 (float, int8) rescoring took {(time.time()-start_time)*1000:.2f} ms")

        return hits
 

    def __len__(self):
        return self.index.ntotal

 