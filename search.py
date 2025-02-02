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
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
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

    def search(self, query_text, is_company_search=False, max_results=50):
        """Perform search with different mechanisms for company vs technology search"""
        try:
            current_year = datetime.now().year

            if is_company_search:
                # Company search - scroll through all awards
                results = self.client.scroll(
                    collection_name="sbir_awards",
                    with_payload=True,
                    limit=10000
                )[0]
                
                processed_results = []
                
                # Multiple variations for matching
                query_cleaned = query_text.lower()
                query_words = query_cleaned.split()
                
                for r in results:
                    # Get original company name and various matching variants
                    company_name = r.payload['company']
                    company_lower = company_name.lower()
                    
                    # Comprehensive matching conditions
                    match_conditions = [
                        # Exact match
                        query_cleaned == company_lower,
                        
                        # Contains full query
                        query_cleaned in company_lower,
                        
                        # Matches any word in the company name
                        any(word in company_lower for word in query_words),
                        
                        # Partial word matches
                        any(
                            company_lower.startswith(word) or 
                            company_lower.endswith(word) or 
                            word in company_lower
                            for word in query_words
                        )
                    ]
                    
                    # If any match condition is true
                    if any(match_conditions):
                        # Calculate recency factor (10% of score)
                        year = r.payload['award_year']
                        recency_factor = max(0, 1 - (current_year - year) * 0.05)
                        
                        # Combine match relevance with recency
                        # 90% match relevance, 10% recency
                        relevance_score = 1.0  # Default for matched results
                        final_score = (0.9 * relevance_score) + (0.1 * recency_factor)
                        
                        processed_results.append({
                            'score': final_score,
                            'similarity': relevance_score,
                            'year': year,
                            'company': company_name,
                            'title': r.payload['award_title'],
                            'abstract': r.payload['abstract'],
                            'amount': r.payload['award_amount'],
                            'contract': r.payload['contract'],
                            'branch': r.payload['branch']
                        })
                
                # Sort results by final score (descending)
                processed_results.sort(key=lambda x: x['score'], reverse=True)
                
                return processed_results[:max_results]
            
            # Technology/topic search
            else:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    semantic_results = self.client.search(
                        collection_name="sbir_awards",
                        query_vector=self.model.encode(query_text, normalize_embeddings=True).tolist(),
                        limit=max_results * 2 if max_results else 10000
                    )
                
                processed_results = []
                for r in semantic_results:
                    # Calculate recency factor (10% of score)
                    year = r.payload['award_year']
                    recency_factor = max(0, 1 - (current_year - year) * 0.05)
                    
                    # Combine semantic similarity with recency
                    # 90% semantic score, 10% recency
                    final_score = (0.9 * r.score) + (0.1 * recency_factor)
                    
                    processed_results.append({
                        'score': final_score,
                        'similarity': r.score,
                        'year': year,
                        'company': r.payload['company'],
                        'title': r.payload['award_title'],
                        'abstract': r.payload['abstract'],
                        'amount': r.payload['award_amount'],
                        'contract': r.payload['contract'],
                        'branch': r.payload['branch']
                    })
                
                # Sort by final score
                processed_results.sort(key=lambda x: x['score'], reverse=True)
                return processed_results[:max_results]

        except Exception as e:
            print(f"Error during search: {str(e)}")
            traceback.print_exc()
            return []

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
