import chromadb

# 1. Connect to ChromaDB (The Brain)
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
collection = chroma_client.get_collection(name="legal_docs")

# 2. The User's Question
query_text = "What was the verdict in the Sharma case?"
print(f"🤖 User asks: '{query_text}'")

# 3. Ask the Database
# query_texts: The question to convert to math
# n_results: How many matching chunks to find
results = collection.query(
    query_texts=[query_text],
    n_results=1
)

# 4. Show the Answer
print("\n🔍 AI Found this Document:")
print("-" * 40)
print(results['documents'][0][0])
print("-" * 40)