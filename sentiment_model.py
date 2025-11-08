import pandas as pd
import numpy as np
import re
from typing import Dict, Tuple, List
import csv
import os # Added to check if file exists

# --- Constants ---
EMBEDDING_FILE = 'mock_embeddings.txt'
COMMENTS_FILE = 'comments.csv'
NEUTRALITY_THRESHOLD = 0.1  # (tau)

class SentimentModel:
    """
    This class encapsulates all the logic from your project proposal.
    It does not interact with the user directly (no input/print).
    """
    def __init__(self):
        print("--- Sentiment-based Website Satisfaction Index ---")
        print("Loading model... This may take a moment.")
        
        # Load embeddings and calculate reference vectors once on startup
        self.embeddings, self.embedding_dim = self._load_embeddings(EMBEDDING_FILE)
        self.s_pos, self.s_neg = self._calculate_reference_vectors()
        
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        sim = self.cosine_similarity(self.s_pos, self.s_neg)
        print(f"Similarity between s_pos and s_neg: {sim:.4f}")

    def _load_embeddings(self, file_path: str) -> Tuple[Dict[str, np.ndarray], int]:
        embeddings_index = {}
        embedding_dim = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    
                    word = parts[0]
                    try:
                        values = np.array(parts[1:], dtype='float32')
                        embeddings_index[word] = values
                        if embedding_dim == 0:
                            embedding_dim = len(values)
                    except ValueError:
                        continue # Skip non-numeric lines

            if embedding_dim == 0:
                raise ValueError("No valid embedding vectors found in file.")
                
            return embeddings_index, embedding_dim

        except FileNotFoundError:
            print(f"Error: Embedding file not found at {file_path}")
            exit(1)

    def _calculate_reference_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        POSITIVE_SEED_WORDS = ['good', 'great', 'love', 'excellent', 'best', 'awesome', 'happy']
        NEGATIVE_SEED_WORDS = ['bad', 'terrible', 'hate', 'worst', 'awful', 'poor', 'sad']
        
        def get_average_vector(words: List[str]) -> np.ndarray:
            vector_sum = np.zeros(self.embedding_dim)
            word_count = 0
            for word in words:
                if word in self.embeddings:
                    vector_sum += self.embeddings[word]
                    word_count += 1
            
            if word_count == 0:
                print(f"Warning: No seed words found for list: {words}.")
                return np.zeros(self.embedding_dim)
                
            return vector_sum / word_count

        s_pos = get_average_vector(POSITIVE_SEED_WORDS)
        s_neg = get_average_vector(NEGATIVE_SEED_WORDS)
        return s_pos, s_neg

    def clean_text(self, text: str) -> List[str]:
        if not isinstance(text, str):
            return []
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        words = text.split()
        return words

    def create_document_vector(self, words: List[str]) -> np.ndarray:
        d = np.zeros(self.embedding_dim)
        word_count = 0
        for word in words:
            if word in self.embeddings:
                d += self.embeddings[word]
                word_count += 1
        
        if word_count > 0:
            d = d / word_count
        return d

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        return dot_product / (norm_v1 * norm_v2)

    def classify_sentiment(self, d: np.ndarray) -> Tuple[str, float, float]:
        if np.linalg.norm(d) == 0:
            return "Neutral", 0.0, 0.0
            
        sim_pos = self.cosine_similarity(d, self.s_pos)
        sim_neg = self.cosine_similarity(d, self.s_neg)
        
        if sim_pos > sim_neg + NEUTRALITY_THRESHOLD:
            return "Positive", sim_pos, sim_neg
        elif sim_neg > sim_pos + NEUTRALITY_THRESHOLD:
            return "Negative", sim_pos, sim_neg
        else:
            return "Neutral", sim_pos, sim_neg

    def calculate_wsi(self, counts: Dict[str, int]) -> Tuple[float, int]:
        count_pos = counts.get('Positive', 0)
        count_neu = counts.get('Neutral', 0)
        count_neg = counts.get('Negative', 0)
        count_total = count_pos + count_neu + count_neg
        
        if count_total == 0:
            return 0.0, 0
            
        numerator = (count_pos * 1) + (count_neg * -1)
        wsi = (numerator / count_total) * 100.0
        return wsi, count_total

    def get_available_sites(self) -> List[str]:
        """Reads the CSV and returns a list of unique website IDs."""
        try:
            # FIX: Tell pandas to read empty strings as "" not NaN
            df = pd.read_csv(COMMENTS_FILE, keep_default_na=False)
            return sorted(df['Website_ID'].unique())
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"Error reading sites from CSV: {e}")
            return []

    def get_website_analysis(self, target_site_id: str) -> Dict:
        """
        Analyzes a site and returns a dictionary of results
        instead of printing to console.
        """
        try:
            # FIX: Tell pandas to read empty strings as "" not NaN
            df = pd.read_csv(COMMENTS_FILE, keep_default_na=False)
        except FileNotFoundError:
            return {"error": f"Comments file not found: {COMMENTS_FILE}"}
        except Exception as e:
            return {"error": f"Error reading CSV: {e}"}

        df_site = df[df['Website_ID'] == target_site_id].copy()

        # --- FIX ---
        # Filter out any rows that have an empty string for the comment.
        # This stops the blank "ghost" comment from being analyzed.
        df_site = df_site[df_site['User_Comment'] != ""]
        # --- END FIX ---

        if df_site.empty:
            return {"error": f"No valid (non-blank) comments found for Website_ID '{target_site_id}'."}

        results = []
        for comment in df_site['User_Comment']:
            words = self.clean_text(comment)
            d = self.create_document_vector(words)
            label, sim_pos, sim_neg = self.classify_sentiment(d)
            results.append((label, sim_pos, sim_neg))

        df_site['Sentiment'] = [r[0] for r in results]
        df_site['Sim_Positive'] = [r[1] for r in results]
        df_site['Sim_Negative'] = [r[2] for r in results]

        sentiment_counts = df_site['Sentiment'].value_counts().to_dict()
        wsi, total_comments = self.calculate_wsi(sentiment_counts)
        
        # Return a structured dictionary
        return {
            "wsi": wsi,
            "total_comments": total_comments,
            "counts": sentiment_counts,
            "labeled_dataframe": df_site
        }

    def add_comment_to_csv(self, website_id: str, user_comment: str) -> str:
        """
        Appends a new comment to the CSV file.
        This is modified from your original 'add_comment' function.
        """
        # Allow empty comments, but not an empty site ID
        if not website_id:
            return "Error: Please provide a Website ID."

        # --- FIX for pandas escaping error ---
        # Let pandas handle the quoting by NOT adding them manually.
        new_data = {'Website_ID': [website_id], 'User_Comment': [user_comment]}
        # --- END FIX ---
        
        new_df = pd.DataFrame(new_data)

        try:
            # Check if file exists to determine if we need to write a header
            file_exists = os.path.isfile(COMMENTS_FILE)
            
            with open(COMMENTS_FILE, 'a', newline='', encoding='utf-8') as f:
                # --- FIX for pandas escaping error ---
                # Let pandas use its default quoting (QUOTE_MINIMAL)
                # which will correctly add quotes only when needed (e.g., if a comment has a comma)
                new_df.to_csv(
                    f,
                    header=not file_exists, # Write header only if file is new
                    index=False,
                    quoting=csv.QUOTE_MINIMAL 
                )
                # --- END FIX ---
            return f"Successfully added comment for '{website_id}'."
        except Exception as e:
            return f"Error saving comment: {e}"

    def add_website(self, website_id: str) -> str:
        """
        Adds a new website to the CSV with a blank comment.
        This ensures the site ID appears in the dropdowns.
        """
        if not website_id:
            return "Error: Please provide a Website ID."
            
        # Check if site already exists
        try:
            # FIX: Tell pandas to read empty strings as "" not NaN
            df = pd.read_csv(COMMENTS_FILE, keep_default_na=False)
            if website_id in df['Website_ID'].unique():
                return f"Error: Website '{website_id}' already exists."
        except FileNotFoundError:
            pass # File doesn't exist, so site doesn't exist. That's fine.
        except Exception as e:
            return f"Error checking existing sites: {e}"

        # Add the site with a blank comment
        status = self.add_comment_to_csv(website_id, "")
        if "Successfully" in status:
             return f"Successfully added website '{website_id}'."
        else:
            return status

    # --- BATCH UPLOAD FUNCTION ---
    def batch_add_comments_from_file(self, website_id: str, file_path: str) -> str:
        """
        Reads a .txt file line by line and adds each line as a new comment
        for the specified website_id.
        """
        if not website_id:
            return "Error: Please provide a Website ID."

        # 1. Read all lines from the .txt file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            return f"Error: File not found at {file_path}"
        except Exception as e:
            return f"Error reading file: {e}"
            
        # 2. Clean lines: remove whitespace and skip empty ones
        comments = [line.strip() for line in lines if line.strip()]
        
        if not comments:
            return f"Error: No valid comments found in '{os.path.basename(file_path)}'."
            
        # 3. Prepare data for DataFrame
        # --- FIX for pandas escaping error ---
        # Let pandas handle the quoting by NOT adding them manually.
        website_ids = [website_id] * len(comments)
        df_new = pd.DataFrame({
            'Website_ID': website_ids,
            'User_Comment': comments
        })
        # --- END FIX ---

        # 4. Append the entire DataFrame to the CSV in one go
        try:
            file_exists = os.path.isfile(COMMENTS_FILE)
            with open(COMMENTS_FILE, 'a', newline='', encoding='utf-8') as f:
                # --- FIX for pandas escaping error ---
                # Let pandas use its default quoting (QUOTE_MINIMAL)
                df_new.to_csv( # <--- This line was changed from new_df to df_new
                    f,
                    header=not file_exists, # Write header only if file is new
                    index=False,
                    quoting=csv.QUOTE_MINIMAL
                )
                # --- END FIX ---
            
            return f"Successfully added {len(comments)} comments from '{os.path.basename(file_path)}'."
            
        except Exception as e:
            return f"Error saving batch comments: {e}"