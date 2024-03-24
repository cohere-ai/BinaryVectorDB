"""
This example shows how to create a new database from scratch.

You must have set your Cohere.com API key as environment variable:
export COHERE_API_KEY=your_api_key

It downloads paragraphs from the Simple English Wikipedia dataset. These paragraphs are embedded 
via the Cohere Embed v3 API and then added to the database.
"""
from BinaryVectorDB import CohereBinaryVectorDB
import os
import gzip
import json

simplewiki_file = "simple-wikipedia-example.jsonl.gz"

#If file not exist, download
if not os.path.exists(simplewiki_file):
    cmd = f"wget https://huggingface.co/datasets/Cohere/BinaryVectorDB/resolve/main/simple-wikipedia-example.jsonl.gz"
    os.system(cmd)

# Create the vector DB with an empty folder
# Ensure that you have set your Cohere API key via: export COHERE_API_KEY=<<YOUR_KEY>>
db_folder = "path_to_an_empty_folder/"
db = CohereBinaryVectorDB(db_folder)

if len(db) > 0:
    exit(f"The database {db_folder} is not empty. Please provide an empty folder to create a new database.")

# Read all docs from the jsonl.gz file
docs = []
with gzip.open(simplewiki_file) as fIn:
    for line in fIn:
        docs.append(json.loads(line))

#Limit it to 10k docs to make the next step a bit faster
docs = docs[0:10_000]

# Add all documents to the DB
# docs2text defines a function that maps our documents to a string
# This string is then embedded with the state-of-the-art Cohere embedding model
db.add_documents(doc_ids=list(range(len(docs))), docs, docs2text=lambda doc: doc['title']+" "+doc['text'])


#Now you can search on your db:
query = "Who is the founder of Facebook"
print("Query:", query)
hits = db.search(query, k=3)
for hit in hits:
    print(hit)