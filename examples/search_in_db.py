"""
This is an example how to search in an existing database. 

You must have set your Cohere.com API key as environment variable:
export COHERE_API_KEY=your_api_key

You can find a list of pre-build BinaryVectorDBs here:
https://huggingface.co/datasets/Cohere/BinaryVectorDB

For example, to download the Wikipedia 2023-11-simple database:
wget https://huggingface.co/datasets/Cohere/BinaryVectorDB/resolve/main/wikipedia-2023-11-simple.zip
unzip wikipedia-2023-11-simple.zip

Usage:
python search_in_db.py /path/to/db/folder
"""
from BinaryVectorDB import BinaryVectorDB
import sys 
import logging 

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

db_folder = sys.argv[1]
db = BinaryVectorDB(db_folder)

while True:
    query = input("Query: ")
    hits = db.search(query, k=10)
    for hit in hits:
        print(hit)

    print("\n===================\n")