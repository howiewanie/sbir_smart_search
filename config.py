# config.py
from typing import Dict

# Qdrant Vector Database Configuration
QDRANT_HOST = 'localhost'
QDRANT_PORT = 6333

# Embedding Model Configuration
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'  # Default model

# Indexing Batch Size
BATCH_SIZE = 128  # Adjust based on your system's memory

# Collection Name in Qdrant
COLLECTION_NAME = 'sbir_awards'

# Column Mappings for CSV Parsing
# Customize these to match your specific CSV file's column names
COLUMN_MAPPINGS: Dict[str, str] = {
    'title': 'Award Title',          # Column name for award title
    'abstract': 'Abstract',           # Column name for abstract
    'company': 'Company Name',        # Column name for company
    'agency': 'Awarding Agency',      # Column name for agency
    'phase': 'Award Phase',           # Column name for award phase
    'year': 'Award Year',             # Column name for award year
    'amount': 'Award Amount',         # Column name for award amount
    'topic_code': 'Topic Code',       # Column name for topic code
    'contract': 'Contract Number',    # Column name for contract number
    'branch': 'Funding Branch'        # Column name for funding branch
}

# Optional: Advanced Embedding Configuration
EMBEDDING_CONFIG = {
    'normalize_embeddings': True,     # Normalize vector embeddings
    'show_progress_bar': False,       # Show progress during embedding
    'device': 'auto'                  # 'auto', 'cpu', or 'cuda'
}

# Optional: Search Configuration
SEARCH_CONFIG = {
    'max_results': 50,                # Default max results per search
    'similarity_threshold': 0.5,      # Minimum similarity score to include results
    'use_company_name_search': True   # Enable/disable company name search
}