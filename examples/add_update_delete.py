"""
This example shows how to create a new database from scratch, add documents to it, updates documents, and delete documents.

You must have set your Cohere.com API key as environment variable:
export COHERE_API_KEY=your_api_key

"""
from BinaryVectorDB import BinaryVectorDB
import numpy as np 
import random
import os 
import shutil

#Some tmp folder to create & delete our db
tmp_folder = f"tmp_folder_{random.randint(0, 999_999_999)}/"
os.makedirs(tmp_folder, exist_ok=False)


db = BinaryVectorDB(tmp_folder)
print(f"The DB has currently {len(db)} docs stored")

#### Add some documents  ####
docs = [
    {'_id': 1, 'text': "Alan  Turing  was an English mathematician, computer scientist, logician, cryptanalyst, philosopher and theoretical biologist."},
    {'_id': 2, 'text': 'Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time.'}
]

# Each document needs to have a unique integer as id 
doc_ids = [doc['_id'] for doc in docs]

# Add documents. We pass in the the doc_ids, the docs and a function that extracts the text out of a doc
db.add_documents(doc_ids=doc_ids, docs=docs, docs2text=lambda doc: doc['text'])

print(f"\n\nThe DB has currently {len(db)} docs stored")

query = "Who was Alan Turing"
print("Query:", query)
hits = db.search(query, k=1)
for hit in hits:
    print(hit)

############################
# Add some new documents  
############################
new_docs = [
    {'_id': 3, 'text': 'Maria Curie was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.'}
]

# Each document needs to have a unique integer as id 
new_doc_ids = [doc['_id'] for doc in new_docs]

# Add documents. We pass in the the doc_ids, the docs and a function that extracts the text out of a doc
db.add_documents(doc_ids=new_doc_ids, docs=new_docs, docs2text=lambda doc: doc['text'])

print(f"\n\nThe DB has currently {len(db)} docs stored")

query = "Who was Maria Curie"
print("Query:", query)
hits = db.search(query, k=1)
for hit in hits:
    print(hit)

############################
# Update a document
############################

# To update a document, simple pass in the same id as the document you want to update
new_docs = [
    {'_id': 2, 'text': 'Mark Zuckerberg  is an American businessman and philanthropist.'}
]

# Each document needs to have a unique integer as id 
new_doc_ids = [doc['_id'] for doc in new_docs]

# Add documents. We pass in the the doc_ids, the docs and a function that extracts the text out of a doc
db.add_documents(doc_ids=new_doc_ids, docs=new_docs, docs2text=lambda doc: doc['text'])

print(f"\n\nThe DB has currently {len(db)} docs stored")

query = "Who is Mark Zuckerberg"
print("Query:", query)
hits = db.search(query, k=1)
for hit in hits:
    print(hit)


############################
# Remove a document
############################

# Pass in the ID you want to remove
db.remove_doc(2)

print(f"\n\nThe DB has currently {len(db)} docs stored")

query = "Who is Mark Zuckerberg"
print("Query:", query)
hits = db.search(query, k=3)
for hit in hits:
    print(hit)


