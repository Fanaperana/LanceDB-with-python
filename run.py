from textsplitter import TextSplitter
import lancedb
from lancedb.pydantic import LanceModel, Vector
import pyarrow as pa
import pandas as pd
import ollama
import numpy as np
from pydantic import BaseModel
import nltk

# Download NLTK stopwords and punkt tokenizer
nltk.download('stopwords')
nltk.download('punkt')

# Define the schema
class LanceSchema(LanceModel):
    id: str
    vector: Vector(768)  # Specify the dimension here
    payload: str

# Sample text from a text file
with open('./knowledge/rust.txt', 'r', encoding='utf-8') as file:
    sample_text = file.read()

# Step 1: Split the text
text_splitter = TextSplitter(max_token_size=100, end_sentence=True, preserve_formatting=True,
                             remove_urls=True, replace_entities=True, remove_stopwords=True, language='english')

chunks = text_splitter.split_text(sample_text)

# Initialize the database connection and table
uri = 'lancedb/data'
db = lancedb.connect(uri)

# Step 2: Generate embeddings for each chunk and add to the table
data = []
for i, chunk in enumerate(chunks):
    embedding = ollama.embeddings(model='nomic-embed-text', prompt=chunk)
    
    # Debugging: Print the structure of the embedding response
    # print(f"Embedding response for chunk {i + 1}:\n{embedding}\n")
    
    # Check if the embedding contains the vector
    if 'embedding' in embedding:
        vector = embedding['embedding']  # Using the correct key 'embedding'
    else:
        print(f"Error: 'embedding' key not found in the embedding response for chunk {i + 1}.")
        continue

    record = LanceSchema(
        id=f"chunk{i}",
        vector=np.array(vector, dtype=np.float32).tolist(),  # Convert to list
        payload=chunk
    )

    data.append(record)

print("Data added.")

# Create the table with the data
tbl = db.create_table("documents", data=data, exist_ok=False)

# Function to generate embedding for a query and search the table
def search_query(query):
    embedding = ollama.embeddings(model='nomic-embed-text', prompt=query)
    
    # Debugging: Print the structure of the embedding response
    # print(f"Embedding response for query:\n{embedding}\n")
    print("Data Embedding for search")
    
    # Check if the embedding contains the vector
    if 'embedding' in embedding:
        vector = embedding['embedding']
    else:
        print("Error: 'embedding' key not found in the embedding response for the query.")
        return []

    tb = db.open_table("documents")
    results = tb.search(vector) \
        .metric("cosine") \
        .limit(10) \
        .to_list()

    return results

# Get user input and search the database
user_query = input("Enter your query: ")
search_results = search_query(user_query)

# Print the results
for i, result in enumerate(search_results):
    print(f"Result {i + 1}: {result['payload']}")
