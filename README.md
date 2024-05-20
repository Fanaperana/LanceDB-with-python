# Step-by-Step Guide for Integrating Embeddings with LanceDB

This guide provides a step-by-step process for integrating text embeddings into LanceDB and using them as a knowledge base for LLMs. Follow these steps to ensure proper setup and execution.

## Prerequisites

1. Install the necessary libraries:
```bash
pip install lancedb pydantic pyarrow pandas ollama numpy nltk
```
For verctor db full text search you will need to use:
```bash
pip install tantivy==0.20.1
```

2. Download NLTK stopwords and the Punkt tokenizer:
```py
import nltk
nltk.download('stopwords')
nltk.download('punkt')

```

### Step 1: Define the Schema
Create a schema using `LanceModel` from `lancedb.pydantic` to define the structure of your data:

```py
from lancedb.pydantic import LanceModel, Vector

class LanceSchema(LanceModel):
    id: str
    vector: Vector(768)  # Specify the dimension here
    payload: str

```

### Step 2: Read the Text File
Load the text data from a file:

```py
text_file_path = './knowledge/rust.txt' # the path of your knowledge document
with open(text_file_path, 'r', encoding='utf-8') as file:
    sample_text = file.read()
```

### Step 3: Split the Text
Use `TextSplitter` to split the text into manageable chunks:

```py
from textsplitter import TextSplitter

text_splitter = TextSplitter(
    max_token_size=100, 
    end_sentence=True, 
    preserve_formatting=True,
    remove_urls=True, 
    replace_entities=True, 
    remove_stopwords=True, 
    language='english'
)

chunks = text_splitter.split_text(sample_text)
```

### Step 4: Initialize the Database
Connect to the LanceDB and initialize the table:

```py
import lancedb

uri = 'lancedb/data'
db = lancedb.connect(uri)
table_name = "documents"
table_exists = table_name in db.table_names()
```

### Step 5: Generate Embeddings and Populate the Table
Generate embeddings for each text chunk using Ollama and populate the table:

```py
import ollama
import numpy as np

if not table_exists:
    data = []
    for i, chunk in enumerate(chunks):
        embedding = ollama.embeddings(model='nomic-embed-text', prompt=chunk)
        
        # Debugging: Print the structure of the embedding response
        print(f"Embedding response for chunk {i + 1}:\n{embedding}\n")
        
        # Check if the embedding contains the vector
        if 'embedding' in embedding:
            vector = embedding['embedding']
        else:
            print(f"Error: 'embedding' key not found in the embedding response for chunk {i + 1}.")
            continue

        record = LanceSchema(
            id=f"chunk{i}",
            vector=np.array(vector, dtype=np.float32).tolist(),  # Convert to list
            payload=chunk
        )

        data.append(record)

    tbl = db.create_table(table_name, data=data)
else:
    print(f"Table '{table_name}' already exists. Skipping creation.")
```

### Step 6: Create Full-Text Search Index (FTS)
Ensure the FTS index is created for efficient searching:

```py
def create_fts_index_if_not_exists(tb):
    try:
        tb.create_fts_index("payload")
    except ValueError as e:
        if "Index already exists" in str(e):
            print("FTS index already exists. Skipping creation.")
        else:
            raise e
```

### Step 7: Search the Database
Define a function to generate embeddings for a query and search the table:

```py
def search_query(query):
    tb = db.open_table(table_name)
    create_fts_index_if_not_exists(tb)
    
    results = tb.search(query) \
        .limit(10) \
        .select(["payload"]) \
        .to_list()

    return results
```

### Step 8: User Input and Display Results
Get the user's query, search the database, and display the results:

```py
user_query = input("Enter your query: ")
search_results = search_query(user_query)

# Print the results
for i, result in enumerate(search_results):
    print(f"\nResult {i + 1}: {result['payload']}\n")

```

## Summary
This guide covers the setup and execution of integrating text embeddings into LanceDB, creating a full-text search index, and querying the database. Make sure to follow each step carefully to ensure correct implementation.

