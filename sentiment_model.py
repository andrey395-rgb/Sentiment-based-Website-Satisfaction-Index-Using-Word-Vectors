import pandas as pd
import numpy as np
import re
from typing import Dict, Tuple, List
import csv
import os

# --- Constants ---
EMBEDDING_FILE = 'mock_embeddings.txt'
COMMENTS_FILE = 'comments.csv'
NEUTRALITY_THRESHOLD = 0.1  # Threshold (tau) for sentiment classification

class SentimentModel:
    """
    Sentiment Analysis Model using Word Vector Embeddings.
    
    This class implements sentiment analysis using vector space mathematics:
    - Word embeddings represent words as vectors in a high-dimensional space
    - Reference vectors represent positive and negative sentiment directions
    - Document vectors are created via linear combination of word vectors
    - Cosine similarity measures semantic alignment between vectors
    """
    def __init__(self):
        print("--- Sentiment-based Website Satisfaction Index ---")
        print("Loading model... This may take a moment.")
        
        # Load word embeddings (vectors in embedding_dim-dimensional space)
        self.embeddings, self.embedding_dim = self._load_embeddings(EMBEDDING_FILE)
        
        # Calculate reference vectors for positive and negative sentiment
        self.s_pos, self.s_neg = self._calculate_reference_vectors()
        
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        sim = self.cosine_similarity(self.s_pos, self.s_neg)
        print(f"Similarity between s_pos and s_neg: {sim:.4f}")

    def _load_embeddings(self, file_path: str) -> Tuple[Dict[str, np.ndarray], int]:
        """
        Loads word embeddings from a text file.
        
        Mathematical Concept: VECTORS AND VECTOR SPACES
        Each word is represented as a vector in an n-dimensional vector space.
        The embedding_dim defines the dimensionality of this space (typically 50-300).
        Each dimension captures a semantic feature of the word.
        
        Returns:
            Dictionary mapping words to their vector representations (numpy arrays)
            The dimensionality of the embedding space
        """
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
                        continue

            if embedding_dim == 0:
                raise ValueError("No valid embedding vectors found in file.")
                
            return embeddings_index, embedding_dim

        except FileNotFoundError:
            print(f"Error: Embedding file not found at {file_path}")
            exit(1)

    def _calculate_reference_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates reference vectors for positive and negative sentiment.
        
        Mathematical Concept: LINEAR COMBINATION
        Reference vectors are computed as the arithmetic mean of seed word vectors.
        This creates a centroid vector that represents the semantic direction of
        sentiment. For seed words w1, w2, ..., wn with vectors v1, v2, ..., vn:
        
        s = (1/n) * (v1 + v2 + ... + vn)
        
        This linear combination gives us a reference point in vector space that
        represents the "average" semantic meaning of positive/negative sentiment.
        
        Returns:
            Tuple of (s_pos, s_neg) - positive and negative reference vectors
        """
        POSITIVE_SEED_WORDS = ['good', 'great', 'love', 'excellent', 'best', 'awesome', 'happy']
        NEGATIVE_SEED_WORDS = ['bad', 'terrible', 'hate', 'worst', 'awful', 'poor', 'sad']
        
        def get_average_vector(words: List[str]) -> np.ndarray:
            """
            Computes the average vector from a list of seed words.
            This is a weighted linear combination with equal weights (1/n).
            """
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
        """Preprocesses text by lowercasing and removing punctuation."""
        if not isinstance(text, str):
            return []
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        return words

    def create_document_vector(self, words: List[str]) -> np.ndarray:
        """
        Creates a document vector by combining word vectors.
        
        Mathematical Concept: LINEAR COMBINATION
        The document vector d is computed as the average of all word vectors
        in the document. For words w1, w2, ..., wn with vectors v1, v2, ..., vn:
        
        d = (1/n) * (v1 + v2 + ... + vn)
        
        This represents the document as a single point in the vector space,
        capturing the aggregate semantic meaning of all words in the document.
        The averaging operation ensures documents of different lengths are
        comparable by normalizing the contribution of each word.
        
        Returns:
            Document vector d in the same embedding space as word vectors
        """
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
        """
        Calculates the cosine similarity between two vectors.
        
        Mathematical Concepts:
        1. INNER PRODUCT (DOT PRODUCT): v1 · v2 = Σ(v1_i * v2_i)
           Measures the alignment between two vectors' directions.
        
        2. VECTOR NORM: ||v|| = √(v · v) = √(Σ(v_i²))
           Measures the magnitude (length) of a vector in the vector space.
        
        3. COSINE SIMILARITY: cos(θ) = (v1 · v2) / (||v1|| * ||v2||)
           Measures the cosine of the angle between two vectors.
           - Returns 1 if vectors point in the same direction (max similarity)
           - Returns 0 if vectors are orthogonal (no similarity)
           - Returns -1 if vectors point in opposite directions (max dissimilarity)
        
        Cosine similarity is preferred over dot product for semantic similarity
        because it is normalized by vector magnitude, making it scale-invariant.
        This allows comparison of vectors regardless of their individual magnitudes.
        
        Returns:
            Cosine similarity value in range [-1, 1]
        """
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        return dot_product / (norm_v1 * norm_v2)

    def classify_sentiment(self, d: np.ndarray) -> Tuple[str, float, float]:
        """
        Classifies the sentiment of a document vector.
        
        Mathematical Concepts:
        1. VECTOR NORM: Checks if document vector is zero vector (||d|| = 0)
           Zero vectors cannot be meaningfully compared via cosine similarity.
        
        2. COSINE SIMILARITY: Compares document vector d with reference vectors
           s_pos and s_neg to determine semantic alignment.
           - Higher similarity to s_pos indicates positive sentiment
           - Higher similarity to s_neg indicates negative sentiment
           - Similar similarities indicate neutral sentiment
        
        3. DECISION BOUNDARY: Uses threshold τ (NEUTRALITY_THRESHOLD) to create
           a margin between positive and negative classifications. This prevents
           documents with very similar positive/negative scores from being
           misclassified due to noise.
        
        Classification rule:
        - Positive if: sim(d, s_pos) > sim(d, s_neg) + τ
        - Negative if: sim(d, s_neg) > sim(d, s_pos) + τ
        - Neutral otherwise
        
        Returns:
            Tuple of (sentiment_label, similarity_to_positive, similarity_to_negative)
        """
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
        """
        Calculates the Website Satisfaction Index (WSI).
        
        Mathematical Concept: LINEAR COMBINATION
        WSI is computed as a weighted average of sentiment counts:
        
        WSI = ((count_pos * 1) + (count_neg * -1)) / count_total * 100
        
        This is a linear combination where:
        - Positive comments contribute +1
        - Negative comments contribute -1
        - Neutral comments contribute 0 (not included in numerator)
        
        The result is normalized by total count and scaled to [-100, 100] range:
        - +100 indicates all comments are positive
        - -100 indicates all comments are negative
        - 0 indicates balanced sentiment or all neutral
        
        Returns:
            Tuple of (WSI score, total comment count)
        """
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
            df = pd.read_csv(COMMENTS_FILE, keep_default_na=False)
            return sorted(df['Website_ID'].unique())
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"Error reading sites from CSV: {e}")
            return []

    def get_website_analysis(self, target_site_id: str) -> Dict:
        """
        Analyzes all comments for a website and returns comprehensive results.
        
        Processing pipeline:
        1. Load comments from CSV for the target website
        2. For each comment:
           a. Clean and tokenize text
           b. Create document vector (linear combination of word vectors)
           c. Classify sentiment using cosine similarity to reference vectors
        3. Calculate WSI score (linear combination of sentiment counts)
        
        Returns:
            Dictionary containing WSI score, counts, labeled dataframe, and summary text
        """
        try:
            df = pd.read_csv(COMMENTS_FILE, keep_default_na=False)
        except FileNotFoundError:
            return {"error": f"Comments file not found: {COMMENTS_FILE}"}
        except Exception as e:
            return {"error": f"Error reading CSV: {e}"}

        df_site = df[df['Website_ID'] == target_site_id].copy()
        df_site = df_site[df_site['User_Comment'] != ""]

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
        
        summary_content = (
            f"Website ID: {target_site_id}\n"
            f"WSI: {wsi:.2f}\n"
            f"Total Comments: {total_comments}\n"
            f"Positive Comments: {sentiment_counts.get('Positive', 0)}\n"
            f"Neutral Comments: {sentiment_counts.get('Neutral', 0)}\n"
            f"Negative Comments: {sentiment_counts.get('Negative', 0)}\n"
        )
        
        return {
            "wsi": wsi,
            "total_comments": total_comments,
            "counts": sentiment_counts,
            "labeled_dataframe": df_site,
            "summary_content": summary_content
        }

    def add_comment_to_csv(self, website_id: str, user_comment: str) -> str:
        """Appends a new comment to the CSV file."""
        if not website_id:
            return "Error: Please provide a Website ID."

        new_data = {'Website_ID': [website_id], 'User_Comment': [user_comment]}
        new_df = pd.DataFrame(new_data)

        try:
            file_exists = os.path.isfile(COMMENTS_FILE)
            with open(COMMENTS_FILE, 'a', newline='', encoding='utf-8') as f:
                new_df.to_csv(
                    f,
                    header=not file_exists, 
                    index=False,
                    quoting=csv.QUOTE_MINIMAL 
                )
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
            
        try:
            df = pd.read_csv(COMMENTS_FILE, keep_default_na=False)
            if website_id in df['Website_ID'].unique():
                return f"Error: Website '{website_id}' already exists."
        except FileNotFoundError:
            pass 
        except Exception as e:
            return f"Error checking existing sites: {e}"

        status = self.add_comment_to_csv(website_id, "")
        if "Successfully" in status:
             return f"Successfully added website '{website_id}'."
        else:
            return status

    def batch_add_comments_from_file(self, website_id: str, file_path: str) -> str:
        """
        Reads a .txt file line by line and adds each line as a new comment
        for the specified website_id.
        """
        if not website_id:
            return "Error: Please provide a Website ID."

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            return f"Error: File not found at {file_path}"
        except Exception as e:
            return f"Error reading file: {e}"
            
        comments = [line.strip() for line in lines if line.strip()]
        
        if not comments:
            return f"Error: No valid comments found in '{os.path.basename(file_path)}'."
            
        website_ids = [website_id] * len(comments)
        df_new = pd.DataFrame({
            'Website_ID': website_ids,
            'User_Comment': comments
        })

        try:
            file_exists = os.path.isfile(COMMENTS_FILE)
            with open(COMMENTS_FILE, 'a', newline='', encoding='utf-8') as f:
                df_new.to_csv(
                    f,
                    header=not file_exists,
                    index=False,
                    quoting=csv.QUOTE_MINIMAL
                )
            
            return f"Successfully added {len(comments)} comments from '{os.path.basename(file_path)}'."
            
        except Exception as e:
            return f"Error saving batch comments: {e}"