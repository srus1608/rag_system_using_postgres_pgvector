import os
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

class DatabaseManager:
    def __init__(self):
        # Connect to your PostgreSQL database using environment variables
        self.conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASS'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT', '5432')
        )
        self.create_tables()

    def create_tables(self):
        with self.conn.cursor() as cur:
            # Enable vector extension if not already enabled
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create the embeddings table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    text_id VARCHAR(255),
                    text_content TEXT,
                    embedding vector(384)
                );
            """)
            self.conn.commit()

    def add_embedding_to_db(self, embedding, text_id, text_content):
        with self.conn.cursor() as cur:
            # Check if embedding is numpy array or list
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            cur.execute(
                "INSERT INTO embeddings (text_id, text_content, embedding) VALUES (%s, %s, %s)",
                (text_id, text_content, embedding_list)
            )
            self.conn.commit()

    def search_similar_vectors(self, query_embedding, top_k=2):
        with self.conn.cursor() as cur:
            # Check if query_embedding is numpy array or list
            query_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            cur.execute("""
                SELECT text_content, 1 - (embedding <=> %s::vector) as similarity
                FROM embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_list, query_list, top_k))
            
            results = cur.fetchall()
            # Format results to match the expected format (text, score)
            return [(text, float(score)) for text, score in results]

    def clear_embeddings(self):
        """Clear all existing embeddings from the database"""
        with self.conn.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE embeddings")
        self.conn.commit()

    def close(self):
        self.conn.close()
