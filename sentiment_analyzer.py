import pandas as pd
import numpy as np
import re
from typing import Dict, Tuple, List
import csv

# --- Constants based on your proposal ---

# 1. File Paths (Update these to your real files)
EMBEDDING_FILE = 'mock_embeddings.txt'
COMMENTS_FILE = 'comments.csv'

# 2. Model Parameters
# The Neutrality Threshold (tau from section 3.2)
# This is the "buffer" to prevent tiny differences from causing a classification.
# You will need to "tune" this value based on your results.
NEUTRALITY_THRESHOLD = 0.1  # (tau)

# --- Phase 1: Setup and Processing ---

def load_embeddings(file_path: str) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Loads word embeddings from a text file.
    Expected format: word val1 val2 val3 ...
    
    Returns:
        A dictionary mapping words to their numpy vector.
        The dimension of the embeddings (n).
    """
    print(f"Loading word embeddings from {file_path}...")
    embeddings_index = {}
    embedding_dim = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                parts = line.split()
                if len(parts) < 2:
                    print(f"Warning: Skipping line {i+1}: format error.")
                    continue
                
                word = parts[0]
                try:
                    values = np.array(parts[1:], dtype='float32')
                    embeddings_index[word] = values
                    # Set embedding dimension from the first valid line
                    if embedding_dim == 0:
                        embedding_dim = len(values)
                except ValueError:
                    print(f"Warning: Skipping word '{word}': non-numeric vector.")

        if embedding_dim == 0:
            raise ValueError("No valid embedding vectors found in file.")
            
        print(f"Loaded {len(embeddings_index)} word vectors.")
        print(f"Embedding dimension (n): {embedding_dim}")
        return embeddings_index, embedding_dim

    except FileNotFoundError:
        print(f"Error: Embedding file not found at {file_path}")
        print("Please download a pre-trained model (like GloVe) or use the mock_embeddings.txt file.")
        exit(1)
    except Exception as e:
        print(f"An error occurred during embedding load: {e}")
        exit(1)

def calculate_reference_vectors(embeddings: Dict[str, np.ndarray], embedding_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates s_pos and s_neg by averaging seed word vectors.
    (As described in section 4.2)
    """
    # Define your seed words here.
    # These MUST exist in your embedding file.
    POSITIVE_SEED_WORDS = ['good', 'great', 'love', 'excellent', 'best', 'awesome', 'happy']
    NEGATIVE_SEED_WORDS = ['bad', 'terrible', 'hate', 'worst', 'awful', 'poor', 'sad']
    
    def get_average_vector(words: List[str]) -> np.ndarray:
        vector_sum = np.zeros(embedding_dim)
        word_count = 0
        for word in words:
            if word in embeddings:
                vector_sum += embeddings[word]
                word_count += 1
            else:
                print(f"Warning: Seed word '{word}' not in embedding file.")
        
        if word_count == 0:
            print(f"Error: No seed words found for list: {words}. Cannot create reference vector.")
            # Return a zero vector to avoid crashing, but this is a critical error.
            return np.zeros(embedding_dim)
            
        return vector_sum / word_count

    print("Calculating reference sentiment vectors (s_pos, s_neg)...")
    s_pos = get_average_vector(POSITIVE_SEED_WORDS)
    s_neg = get_average_vector(NEGATIVE_SEED_WORDS)
    
    # Per your testing plan (4.4), check orthogonality.
    # A value near 0 means they are orthogonal (point in different directions).
    # A value near 1 means they are very similar (bad).
    # A value near -1 means they are perfectly opposite (good).
    similarity = cosine_similarity(s_pos, s_neg)
    print(f"Similarity between s_pos and s_neg: {similarity:.4f}")
    if similarity > 0.5:
        print("Warning: Your positive and negative seed words are very similar! Check your words or embeddings.")
        
    return s_pos, s_neg

def clean_text(text: str) -> List[str]:
    """
    Basic text cleaning: lowercase, remove punctuation, split into words.
    (As described in section 4.2)
    
    TODO: Enhance this with stop-word removal and stemming/lemmatization
    using a library like NLTK.
    """
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()
    # Placeholder for stop-word removal
    # from nltk.corpus import stopwords
    # stop_words = set(stopwords.words('english'))
    # words = [w for w in words if not w in stop_words]
    return words

# --- Phase 2: Comment Classification ---

def create_document_vector(words: List[str], embeddings: Dict[str, np.ndarray], embedding_dim: int) -> np.ndarray:
    """
    Creates the Document Vector (d) by averaging its word vectors.
    (Formula 3.2: d = (1/|Words|) * sum(w_i))
    """
    d = np.zeros(embedding_dim)
    word_count = 0
    for word in words:
        if word in embeddings:
            d += embeddings[word]
            word_count += 1
    
    if word_count > 0:
        # Calculate the average (centroid) vector
        d = d / word_count
        
    # If word_count is 0, 'd' will be np.zeros(embedding_dim)
    return d

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two vectors.
    (Formula 3.2: Sim(d,s) = (d . s) / (||d|| ||s||))
    """
    # Use numpy for fast linear algebra
    dot_product = np.dot(v1, v2)
    
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Check for zero vectors to avoid division by zero
    # (as per testing plan 4.4)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    return dot_product / (norm_v1 * norm_v2)

def classify_sentiment(d: np.ndarray, s_pos: np.ndarray, s_neg: np.ndarray, tau: float) -> Tuple[str, float, float]:
    """
    Applies the final classification rule from section 3.2.
    
    Returns:
        The sentiment label (str).
        The positive similarity score (float).
        The negative similarity score (float).
    """
    # 1. Handle the case of an empty comment (zero vector)
    if np.linalg.norm(d) == 0:
        return "Neutral", 0.0, 0.0
        
    # 2. Calculate cosine similarities
    sim_pos = cosine_similarity(d, s_pos)
    sim_neg = cosine_similarity(d, s_neg)
    
    # 3. Apply the classification rule
    if sim_pos > sim_neg + tau:
        return "Positive", sim_pos, sim_neg
    elif sim_neg > sim_pos + tau:
        return "Negative", sim_pos, sim_neg
    else:
        return "Neutral", sim_pos, sim_neg

# --- Phase 3: Aggregation and Output ---

def calculate_wsi(counts: Dict[str, int]) -> Tuple[float, int]:
    """
    Calculates the Website Satisfaction Index (WSI) from section 3.2.
    
    Returns:
        The WSI score (float).
        The total count of comments (int).
    """
    count_pos = counts.get('Positive', 0)
    count_neu = counts.get('Neutral', 0)
    count_neg = counts.get('Negative', 0)
    
    count_total = count_pos + count_neu + count_neg
    
    if count_total == 0:
        return 0.0, 0
        
    # WSI = ( (Count_pos * 1) + (Count_neu * 0) + (Count_neg * -1) ) / Count_total
    # And scale to -100 to +100
    
    numerator = (count_pos * 1) + (count_neg * -1)
    wsi = (numerator / count_total) * 100.0
    
    return wsi, count_total

def analyze_website_rating(embeddings: Dict[str, np.ndarray], embedding_dim: int, s_pos: np.ndarray, s_neg: np.ndarray):
    """
    Handles the process of analyzing and displaying the WSI for a chosen website.
    """
    try:
        df = pd.read_csv(COMMENTS_FILE)
    except FileNotFoundError:
        print(f"Error: Comments file not found at {COMMENTS_FILE}")
        print("You can add a comment first to create the file.")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    available_sites = df['Website_ID'].unique()
    if len(available_sites) == 0:
        print("No websites found in comments file. Please add a comment first.")
        return

    print("\nAvailable Website IDs:")
    print(available_sites)
    target_site_id = input("Please enter the Website_ID to analyze: ")

    df_site = df[df['Website_ID'] == target_site_id].copy()

    if df_site.empty:
        print(f"Error: No comments found for Website_ID '{target_site_id}'.")
        return

    print(f"\nAnalyzing {len(df_site)} comments for '{target_site_id}'...")

    results = []
    for comment in df_site['User_Comment']:
        words = clean_text(comment)
        d = create_document_vector(words, embeddings, embedding_dim)
        label, sim_pos, sim_neg = classify_sentiment(d, s_pos, s_neg, NEUTRALITY_THRESHOLD)
        results.append((label, sim_pos, sim_neg))

    df_site['Sentiment'] = [r[0] for r in results]
    df_site['Sim_Positive'] = [r[1] for r in results]
    df_site['Sim_Negative'] = [r[2] for r in results]

    print("Classification complete.")

    sentiment_counts = df_site['Sentiment'].value_counts().to_dict()
    wsi, total_comments = calculate_wsi(sentiment_counts)

    count_pos = sentiment_counts.get('Positive', 0)
    count_neu = sentiment_counts.get('Neutral', 0)
    count_neg = sentiment_counts.get('Negative', 0)

    print("\n--- Analysis Results ---")
    print(f"Website Satisfaction Index (WSI): {wsi:.2f}")
    print("--------------------------")
    print(f"Total Comments Analyzed: {total_comments}")
    if total_comments > 0:
        print(f"Positive Comments: {count_pos} ({count_pos/total_comments:.1%})")
        print(f"Neutral Comments:  {count_neu} ({count_neu/total_comments:.1%})")
        print(f"Negative Comments: {count_neg} ({count_neg/total_comments:.1%})")

    output_filename = f"labeled_output_{target_site_id}.csv"
    try:
        df_site.to_csv(output_filename, index=False)
        print(f"\nSuccessfully saved labeled data to '{output_filename}'")
    except Exception as e:
        print(f"\nError saving output file: {e}")

def add_comment():
    """
    Adds a new comment for a website to the comments CSV file.
    Ensures that only the comment is enclosed in double quotes in the output.
    Example output: Site_A,"good"
    """
    print("\n--- Add a New Comment ---")
    website_id = input("Enter the Website_ID: ").strip()
    user_comment = input("Enter your comment: ").strip()

    # Manually wrap the comment in exactly one pair of double quotes
    user_comment = f'"{user_comment}"'

    new_data = {'Website_ID': [website_id], 'User_Comment': [user_comment]}
    new_df = pd.DataFrame(new_data)

    try:
        with open(COMMENTS_FILE, 'a', newline='', encoding='utf-8') as f:
            # Write header only if file is new/empty
            write_header = f.tell() == 0
            # Disable pandas’ automatic quoting so our quotes stay as-is
            new_df.to_csv(
                f,
                header=write_header,
                index=False,
                quoting=3  # csv.QUOTE_NONE
            )
        print(f"\nSuccessfully added comment for '{website_id}'.")
    except Exception as e:
        print(f"\nError saving comment: {e}")
        
def main():
    """
    Main function to run the complete analysis pipeline with a user menu.
    """
    print("--- Sentiment-based Website Satisfaction Index ---")
    
    embeddings, embedding_dim = load_embeddings(EMBEDDING_FILE)
    s_pos, s_neg = calculate_reference_vectors(embeddings, embedding_dim)
    
    while True:
        print("\n--- Main Menu ---")
        print("1. See the rating of a certain website")
        print("2. Add a comment for a website")
        print("3. Exit")
        choice = input("Please choose an option (1-3): ")

        if choice == '1':
            analyze_website_rating(embeddings, embedding_dim, s_pos, s_neg)
        elif choice == '2':
            add_comment()
        elif choice == '3':
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 3.")

if __name__ == "__main__":
    main()