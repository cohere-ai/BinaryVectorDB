from BinaryVectorDB import CohereBinaryVectorDB

db_folder = "an_existing_db/"
db = CohereBinaryVectorDB(db_folder)

print(f"The DB has currently {len(db)} docs stored")

docs = [
    "BinaryVectorDB is an amazing example how binary & int8 embeddings allows scaling to large datasets",
    "To learn more about BinaryVectorDB visit cohere.com"
]

db.add_documents(ids=list(range(len(docs))), docs=docs, docs2text=lambda doc: doc)

print(f"The DB has currently {len(db)} docs stored")

query = "What is BinaryVectorDB?"
print("Query:", query)
hits = db.search(query, k=3)
for hit in hits:
    print(hit)