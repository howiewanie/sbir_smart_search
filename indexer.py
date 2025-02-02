"""
SBIR Smart Search Indexer
------------------------
A tool for creating searchable vector embeddings from SBIR/STTR award data.
Uses BGE embeddings and Qdrant vector database for efficient similarity search.

Author: Frontier Optic (https://frontieroptic.com)
License: MIT
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import json
import os
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
import torch
import psutil
import time
from typing import Optional, Dict, Any

# Import configuration
from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    MODEL_NAME,
    BATCH_SIZE,
    COLLECTION_NAME,
    COLUMN_MAPPINGS
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sbir_indexing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SBIRIndexer:
    """Vector indexing system for SBIR/STTR awards data."""
    
    def __init__(
        self, 
        batch_size: int = BATCH_SIZE,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        collection_name: str = COLLECTION_NAME,
    ):
        """Initialize the indexing system with configurable parameters."""
        self.batch_size = batch_size
        self.collection_name = collection_name
        self.start_time = None
        
        # Connect to Qdrant
        logger.info(f"Connecting to Qdrant at {host}:{port}...")
        self.client = QdrantClient(host, port=port)
        
        # Load embedding model
        logger.info(f"Loading {MODEL_NAME}...")
        self.model = 
# ===========================
# Choose the language model you want to use and replace 'sentence-transformers/all-MiniLM-L6-v2' if desired.
# The current default model is ultralightweight and gives decent performance for general-purpose searches.
# Examples for alternatives: 'allenai/scibert_scivocab_uncased', 'BAAI/bge-base-en-v1.5'
# ===========================
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        
        # Use GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

    def clean_text(self, text: Any) -> str:
        """Clean and normalize text fields."""
        if pd.isna(text):
            return ""
        return str(text).strip()

    def clean_number(self, value: Any) -> float:
        """Clean and convert numeric values."""
        try:
            if pd.isna(value):
                return 0
            if isinstance(value, str):
                value = value.replace(',', '').replace('$', '')
            return float(value)
        except (ValueError, TypeError):
            return 0

    def clean_year(self, year: Any) -> Optional[int]:
        """Clean and validate year values."""
        try:
            if pd.isna(year):
                return None
            year = int(float(str(year).replace(',', '')))
            return year if 1900 <= year <= 2100 else None
        except (ValueError, TypeError):
            return None

    def prepare_collection(self) -> None:
        """Set up the Qdrant collection with proper schema."""
        try:
            # Get vector size from model
            vector_size = self.model.get_sentence_embedding_dimension()
            
            # Check if collection exists
            collections = self.client.get_collections()
            if any(c.name == self.collection_name for c in collections.collections):
                logger.info("Removing existing collection...")
                self.client.delete_collection(self.collection_name)
            
            # Create new collection
            logger.info("Creating new collection...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            
            # Create indexes for filtering
            logger.info("Creating indices...")
            indexes = [
                ("award_year", models.PayloadSchemaType.INTEGER),
                ("award_amount", models.PayloadSchemaType.FLOAT),
                ("agency", models.PayloadSchemaType.KEYWORD),
                ("phase", models.PayloadSchemaType.KEYWORD),
                ("branch", models.PayloadSchemaType.KEYWORD)
            ]
            
            for field_name, field_type in indexes:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
            
            logger.info("Collection prepared successfully")
            
        except Exception as e:
            logger.error(f"Error preparing collection: {str(e)}")
            raise

    def get_progress_message(self, current: int, total: int, elapsed_time: float) -> str:
        """Generate detailed progress message with estimates."""
        percentage = (current / total) * 100
        
        if current > 0:
            time_per_record = elapsed_time / current
            remaining_records = total - current
            eta = str(timedelta(seconds=int(remaining_records * time_per_record)))
        else:
            eta = "Calculating..."
            
        memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        return (f"Progress: {percentage:.1f}% ({current:,}/{total:,})\n"
                f"Memory usage: {memory:.1f} MB\n"
                f"Time elapsed: {str(timedelta(seconds=int(elapsed_time)))}\n"
                f"Estimated remaining: {eta}")

    def index_data(self, csv_path: str) -> bool:
        """
        Index SBIR/STTR award data from CSV file.
        
        Args:
            csv_path: Path to the CSV file containing award data
            
        Returns:
            bool: True if indexing completed successfully
        """
        try:
            self.start_time = time.time()
            
            logger.info(f"Loading data from {csv_path}")
            df = pd.read_csv(csv_path, low_memory=False)
            total_records = len(df)
            logger.info(f"Found {total_records:,} records to process")
            
            # Process in batches
            with tqdm(total=total_records) as pbar:
                for start_idx in range(0, total_records, self.batch_size):
                    # Get current batch
                    end_idx = min(start_idx + self.batch_size, total_records)
                    batch_df = df.iloc[start_idx:end_idx]
                    
                    # Create embeddings
                    texts = [
                        f"{self.clean_text(row[COLUMN_MAPPINGS['title']])} {self.clean_text(row[COLUMN_MAPPINGS['abstract']])}"
                        for _, row in batch_df.iterrows()
                    ]
                    
                    embeddings = self.model.encode(
                        texts,
                        batch_size=32,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                    
                    # Prepare points
                    points = []
                    for idx, (_, row), embedding in zip(
                        range(start_idx, end_idx),
                        batch_df.iterrows(),
                        embeddings
                    ):
                        point = models.PointStruct(
                            id=idx,
                            vector=embedding.tolist(),
                            payload={
                                'award_title': self.clean_text(row[COLUMN_MAPPINGS['title']]),
                                'abstract': self.clean_text(row[COLUMN_MAPPINGS['abstract']]),
                                'company': self.clean_text(row[COLUMN_MAPPINGS['company']]),
                                'agency': self.clean_text(row[COLUMN_MAPPINGS['agency']]),
                                'phase': self.clean_text(row[COLUMN_MAPPINGS['phase']]),
                                'award_year': self.clean_year(row[COLUMN_MAPPINGS['year']]),
                                'award_amount': self.clean_number(row[COLUMN_MAPPINGS['amount']]),
                                'topic_code': self.clean_text(row[COLUMN_MAPPINGS['topic_code']]),
                                'contract': self.clean_text(row[COLUMN_MAPPINGS['contract']]),
                                'branch': self.clean_text(row[COLUMN_MAPPINGS['branch']])
                            }
                        )
                        points.append(point)
                    
                    # Upload batch
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    
                    # Update progress
                    pbar.update(len(batch_df))
                    
                    # Show detailed progress periodically
                    if end_idx % 1000 == 0 or end_idx == total_records:
                        elapsed = time.time() - self.start_time
                        progress_msg = self.get_progress_message(end_idx, total_records, elapsed)
                        logger.info(f"\n{progress_msg}")
            
            logger.info("\nIndexing completed successfully!")
            return True
            
        except KeyboardInterrupt:
            logger.warning("\n\nIndexing interrupted by user")
            logger.warning(f"Progress saved at record {start_idx:,}")
            return False
            
        except Exception as e:
            logger.error(f"\nError during indexing: {str(e)}")
            return False

def main():
    """Main entry point for the indexing tool."""
    try:
        print("\n=== SBIR Smart Search Indexer ===")
        print("Create searchable vectors from SBIR/STTR award data")
        print("\nFor latest data: https://frontieroptic.com/smartsearch")
        
        # Initialize indexer
        indexer = SBIRIndexer()
        
        # Prepare collection
        print("\nPreparing collection...")
        indexer.prepare_collection()
        
        # Get CSV path
        csv_path = input("\nEnter path to your SBIR CSV file: ").strip()
        if not os.path.exists(csv_path):
            print("Error: File not found!")
            return
        
        print("\nStarting indexing process...")
        print("Progress will be shown as records are processed")
        print("Press Ctrl+C to safely interrupt if needed")
        
        indexer.index_data(csv_path)
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main()