# BinaryVectorDB - Efficient Search on Large Datasets

This repository contains a Binary Vector Database for efficient search on large datasets, aimed for educational purposes.

Most embedding models represent their vectors as float32: These consume a lot of memory and search on these is very slow. At Cohere, we introduced the first embedding model with native [int8 and binary support](https://txt.cohere.com/int8-binary-embeddings/), which give you excellent search quality for a fraction of the cost:


| Model | Search Quality MIRACL | Time to Search 1M docs | Memory Needed 250M Wikipedia Embeddings | Price on AWS (x2gb instance) |
| ----- |:---------------------:|:----------------------:|:---------------------------------------:|:------------:|
| OpenAI text-embedding-3-small | 44.9 | 680 ms | 1431 GB | $65,231 / yr |
| OpenAI text-embedding-3-large | 54.9 | 1240 ms | 2861 GB | $130,463 / yr |
| **Cohere Embed v3 (Multilingual)** | | | | |
| Embed v3 - float32 | 66.3 | 460 ms | 954 GB | $43,488 / yr |
| Embed v3 - binary | 62.8 | 24 ms | 30 GB | $1,359 / yr |
| Embed v3 - binary + int8 rescore | 66.3 | 28 ms | 30 GB memory + 240 GB disk | $1,589 / yr |


# Setup

The setup is easy:
```
pip install BinaryVectorDB
```

To use some of the below examples you need a **Cohere API key** (free or paid) from [https://cohere.com/](Cohere.com). You must set this API key as an environment variable: `export COHERE_API_KEY=your_api_key`  


# Usage - Load an Existing Binary Vector Database

We will talk later how to build your own vector database. For the start, let us use a pre-build binary vector database. We host various pre-build databases on [https://huggingface.co/Cohere/BinaryVectorDB](https://huggingface.co/Cohere/BinaryVectorDB). You can download these and use them localy.

Let us the simple English version from Wikipedia to get started:
```
wget https://huggingface.co/datasets/Cohere/BinaryVectorDB/resolve/main/wikipedia-2023-11-simple.zip
```

And then unzip this file:
```
unzip wikipedia-2023-11-simple.zip
```


## Load the Vector Database

You can load the database easily by pointing it to the unzipped folder from the previous step:

```python
from BinaryVectorDB import BinaryVectorDB

# Point it to the unzipped folder from the previous step
# Ensure that you have set your Cohere API key via: export COHERE_API_KEY=<<YOUR_KEY>>
db = BinaryVectorDB("wikipedia-2023-11-simple/")

query = "Who is the founder of Facebook"
print("Query:", query)
hits = db.search(query)
for hit in hits[0:3]:
    print(hit)
```

The database has 646,424 embeddings and a total size of 962 MB. However, just 80 MB for the binary embeddings are loaded in memory. The documents and their int8 embeddings are kept on disk and are just loaded when needed.

This split of binary embeddings in memory and int8 embeddings & documents on disk allows us to scale to very large datasets without need tons of memory.

# Build your own Binary Vector Database

It is quite easy to build your own Binary Vector Database.

```python
from BinaryVectorDB import BinaryVectorDB
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
db = BinaryVectorDB(db_folder)

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
db.add_documents(doc_ids=list(range(len(docs))), docs=docs, docs2text=lambda doc: doc['title']+" "+doc['text'])
```

The document can be any Python serializable object. You need to provide a function for `docs2text` that map your document to a string. In the above example, we concatenate the title and text field. This string is send to the embedding model to produce the needed text embeddings.



## Updating & Deleting Documents

See [examples/add_update_delete.py](examples/add_update_delete.py) for an example script how to add/update/delete documents in the database.

# Is this a real Vector Database?

Not really. The repository is meant mostly for educational purposes to show techniques how to scale to large datasets. The focus was more on ease of use and some critical aspects are missing in the implementation, like multi-process safety, rollbacks etc. 

If you actually wants to go into production, use a proper vector database like [Vespa.ai](https://blog.vespa.ai/scaling-large-vector-datasets-with-cohere-binary-embeddings-and-vespa/), that allows you to achieve similar results.