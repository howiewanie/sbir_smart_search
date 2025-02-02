import pandas as pd
import unicodedata
import re
import traceback
import sys
import os
import tempfile
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

class SBIRSearch:
    def __init__(self, qdrant_host="localhost", qdrant_port=6333, results_dir=None):
        """Initialize the search system with configurable parameters"""
        try:
            print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}...")
            self.client = QdrantClient(qdrant_host, port=qdrant_port)
            
            print("Loading embedding model...")
            self.model = 
# ===========================
# Choose the language model you want to use and replace 'sentence-transformers/all-MiniLM-L6-v2' if desired.
# The current default model is ultralightweight and gives decent performance for general-purpose searches.
# Examples for alternatives: 'allenai/scibert_scivocab_uncased', 'BAAI/bge-base-en-v1.5'
# ===========================
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

            
            # Make results directory configurable
            self.results_dir = results_dir
            if not self.results_dir:
                self.setup_results_directory()
            
            # Check if collection exists
            collections = self.client.get_collections()
            if not any(c.name == "sbir_awards" for c in collections.collections):
                raise Exception("SBIR awards collection not found. Please run indexing first.")
            
        except Exception as e:
            print(f"Error initializing search system: {str(e)}")
            raise

    def setup_results_directory(self):
        """Create and verify results directory with extensive logging"""
        # List potential directories to try
        potential_dirs = [
            # Try script directory first
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
            # User's home directory
            os.path.join(os.path.expanduser("~"), "SBIR_Results"),
            # Desktop
            os.path.join(os.path.expanduser("~"), "Desktop", "SBIR_Results"),
            # Documents
            os.path.join(os.path.expanduser("~"), "Documents", "SBIR_Results"),
            # Temp directory
            os.path.join(tempfile.gettempdir(), "SBIR_Results")
        ]

        # Print current working directory and script location for debugging
        print(f"Current Working Directory: {os.getcwd()}")
        print(f"Script Location: {os.path.dirname(os.path.abspath(__file__))}")
        print(f"Home Directory: {os.path.expanduser('~')}")
        print(f"Temp Directory: {tempfile.gettempdir()}")

        # Try each potential directory
        for directory in potential_dirs:
            try:
                # Ensure the directory exists
                os.makedirs(directory, exist_ok=True)
                
                # Attempt to write a test file
                test_file = os.path.join(directory, "test_write.txt")
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write("Write test successful")
                
                # Remove test file
                os.remove(test_file)
                
                # If we get here, we've found a writable directory
                self.results_dir = directory
                print(f"Successfully set results directory to: {directory}")
                return
            
            except Exception as e:
                print(f"Could not use directory {directory}: {e}")
        
        # If no directory works, raise an exception
        raise Exception("Could not find a writable directory for results")

    # Rest of the code remains the same as the original script...
    # (Copy the remaining methods clean_text, clean_company_name, find_similar_company, 
    # calculate_score, display_distribution, search, display_results_page, export_results)
    # and the main() function

    def clean_text(self, text):
        """Thoroughly clean and normalize text"""
        if not text or not isinstance(text, str):
            return ""
        
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove non-printable characters
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        
        # Replace problematic quotation marks and other encoding artifacts
        text = text.replace('â€œ', '"').replace('â€', '"')
        
        # Strip leading/trailing whitespace
        return text.strip()

    # Remaining methods would be copied from the original script...

def main():
    try:
        print("\n=== SBIR/STTR Award Search ===")
        print("\nSearch through SBIR/STTR awards from program inception to fiscal year 2023")
        print("\nTips:")
        print("- Enter search terms for technology or company")
        print("- Results are ranked by relevance (90%) and recency (10%)")
        
        # Allow configurable Qdrant host and port
        qdrant_host = os.environ.get('QDRANT_HOST', 'localhost')
        qdrant_port = int(os.environ.get('QDRANT_PORT', 6333))
        
        searcher = SBIRSearch(
            qdrant_host=qdrant_host, 
            qdrant_port=qdrant_port
        )
        
        # Rest of the main function remains the same...
        while True:
            print("\n" + "=" * 80)
            query = input("\nEnter search term (or 'exit'): ").strip()
            if query.lower() == 'exit':
                break

            is_company = input("Are you searching a specific company name? (Enter for no, y for yes): ").lower().startswith('y')
            
            max_input = input("Number of results (Enter for default: 50): ").strip()
            max_results = int(max_input) if max_input else 50

            try:
                results = searcher.search(
                    query_text=query,
                    is_company_search=is_company,
                    max_results=max_results
                )
                
                if results:
                    # Display distribution before showing results
                    searcher.display_distribution(results)
                    
                    # Initialize pagination
                    current_page = 1
                    while True:
                        has_more = searcher.display_results_page(results, current_page)
                        
                        if has_more:
                            action = input("\nOptions: [n]ext page, [d]ownload CSV, or [s]earch again: ").lower()
                            if action == 'n':
                                current_page += 1
                            elif action == 'd':
                                searcher.export_results(results, query)
                                break
                            elif action == 's':
                                break
                        else:
                            action = input("\nOptions: [d]ownload CSV or [s]earch again: ").lower()
                            if action in ['d', 's']:
                                if action == 'd':
                                    searcher.export_results(results, query)
                                break
                else:
                    print("\nNo matching results found.")

            except ValueError as e:
                print(f"\nInvalid input: {str(e)}")
            except Exception as e:
                print(f"\nAn error occurred: {str(e)}")
                traceback.print_exc()

    except Exception as e:
        print(f"\nSystem error: {str(e)}")
        print("Please ensure Qdrant is running and the collection has been indexed.")
        traceback.print_exc()

if __name__ == "__main__":
    main()