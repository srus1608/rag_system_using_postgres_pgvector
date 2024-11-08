from embedding_generator import EmbeddingGenerator
from database import DatabaseManager
import os
from PyPDF2 import PdfReader
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq

# Add Pydantic models
class Metadata(BaseModel):
    similarity_score: float

class RetrievedDocument(BaseModel):
    document_id: str
    content: str
    metadata: Metadata

class RetrievalResult(BaseModel):
    query: str
    retrieved_documents: List[RetrievedDocument]
    retrieval_method: str = "vector search"
    generated_response: Optional[str] = None

DATA_FILE_PATH = 'documents.pdf'
is_pdf = True

def extract_text_from_pdf(file_path):
    """Function to extract text from a PDF file."""
    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text.replace("\x00", "")  # Remove NUL characters if any
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def load_text_samples(file_path, is_pdf=False):
    texts = []
    if is_pdf:
        texts.append(extract_text_from_pdf(file_path))
    else:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file '{file_path}' not found.")
        with open(file_path, 'r') as file:
            texts = list(dict.fromkeys(filter(None, file.read().splitlines())))
    return texts

def generate_augmented_response(query: str, retrieved_items: List[tuple[str, float]]) -> RetrievalResult:
    retrieved_docs = [
        RetrievedDocument(
            document_id=f"doc_{idx}",
            content=text,
            metadata=Metadata(similarity_score=score)
        )
        for idx, (text, score) in enumerate(retrieved_items)
    ]
    
    context = " ".join(text for text, _ in retrieved_items)
    
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    prompt = f"""Answer the following question using ONLY the information provided in the context below. 
    If the context doesn't contain enough information to fully answer the question, 
    respond with what can be answered from the context only. 
    Context: {context} 
    Question: {query}
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[ 
                {"role": "system", "content": """You are an assistant that provides answers strictly based on the given context."""},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=1024
        )

        response = chat_completion.choices[0].message.content.strip()
        
        return RetrievalResult(
            query=query,
            retrieved_documents=retrieved_docs,
            generated_response=response
        )

    except Exception as e:
        print("Error generating response:", str(e))
        return RetrievalResult(
            query=query,
            retrieved_documents=retrieved_docs,
            generated_response="Error generating response."
        )

def main():
    db_manager = DatabaseManager()
    embedding_gen = EmbeddingGenerator()

    # Clear existing data before adding new embeddings
    db_manager.clear_embeddings()

    # Load .pdf document data
    texts = load_text_samples(DATA_FILE_PATH, is_pdf=is_pdf)

    # Add check for duplicate texts
    processed_texts = set()
    for idx, text in enumerate(texts):
        if text in processed_texts:
            continue
        processed_texts.add(text)

        embedding = embedding_gen.generate_embedding(text)
        if embedding is not None:
            db_manager.add_embedding_to_db(embedding, text_id=str(idx), text_content=text)
            print("------------------------------------------------------------------------------------------------")
            print(f"Added embedding for: {text[:50]}...")  # Display first 50 characters for brevity
            print(f"Embeddings for above text: {embedding[:5]}...")  # Display first 5 values for brevity
            print("------------------------------------------------------------------------------------------------")
    
    print("Embeddings added to the database.")
    
    # Loop to continuously prompt user for queries
    while True:
        query_text = input("Enter your query (or type 'exit' to quit): ")
        if query_text.lower() == 'exit':
            print("Exiting...")
            break

        query_embedding = embedding_gen.generate_embedding(query_text)

        print(f"Query Text: {query_text}")
        print(f"Embedding for Query Text: {query_embedding[:5]}...")  # Display first 5 values

        # Search for similar items in the database
        similar_items = db_manager.search_similar_vectors(query_embedding, top_k=3)

        print("Similar items found:")
        for item in similar_items:
            print(item)

        # Generate and print the augmented response
        result = generate_augmented_response(query_text, similar_items)
        # print("Structured Response:")
        # print(result.model_dump_json(indent=2))
        # print("------------------------------------------------------------------------------------------------")
        # print(result)
    db_manager.close()

if __name__ == "__main__":
    main()
