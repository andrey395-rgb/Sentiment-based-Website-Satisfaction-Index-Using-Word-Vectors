Sentiment-based Website Satisfaction Index Using Word Vectors

Project Overview

This project proposes and implements a sentiment analysis system that classifies user feedback into positive, neutral, and negative categories. It uses the mathematical framework of vector spaces and inner products to quantify user sentiment.

By representing textual data (user comments) as numerical vectors, the model uses cosine similarity to measure how closely a comment aligns with pre-defined "positive" and "negative" sentiment vectors. These classifications are then aggregated into a single Website Satisfaction Index (WSI), providing a clear, data-driven metric of overall user satisfaction.

This repository contains two main implementations of this model:

A Python command-line script (sentiment_analyzer.py) for processing local files.

An interactive web application (index.html) that runs the same logic in the browser.

Core Concept

The sentiment of a comment is determined by:

Vectorization: Each word in a comment is converted into a high-dimensional vector using a pre-trained embedding model (e.g., GloVe).

Document Vector: The vectors for all words in a comment are averaged to create a single "document vector" (d) that represents the overall meaning of the comment.

Reference Vectors: A "positive" vector (s_pos) and a "negative" vector (s_neg) are calculated by averaging the vectors of several seed words (e.g., "good", "great" vs. "bad", "terrible").

Cosine Similarity: The model calculates the cosine of the angle between the document vector d and the two reference vectors.

Classification:

If d is directionally closer to s_pos, the comment is "Positive".

If d is directionally closer to s_neg, the comment is "Negative".

If it's in the middle (within a NEUTRALITY_THRESHOLD), it's "Neutral".


Part 1: How to Run the Python Script

This is a command-line tool that reads the comments.csv file, analyzes a specific Website_ID, and saves a new CSV file with the results.

1. Installation

Before running the script, you need to install the pandas and numpy libraries.

pip install pandas numpy


2. Running the Script

Run the script from your terminal inside the project folder.

python sentiment_analyzer.py


The script will:

Load the mock_embeddings.txt and comments.csv files.

Prompt you to enter a Website_ID to analyze (e.g., Site_A).

Print a full analysis and WSI score to the terminal.

Save a new file (e.g., labeled_output_Site_A.csv) with the detailed results.

Part 2: How to Run the Web Application

This is an interactive website that runs the exact same logic entirely in your browser. It allows you to analyze new comments in real-time.

1. Start the Local Server

This web app must be run from a local server. Python has one built-in. From your terminal, run this command inside the project folder:

python -m http.server


You will see a message like Serving HTTP on 0.0.0.0 port 8000.... Leave this terminal running.

2. Access the Web App

Open your web browser (Chrome, Firefox, etc.) and go to the following address:

http://localhost:8000


The index.html file will load automatically.

3. Using the Web App

The page is split into two parts:

Step 1: Analyze a New Comment

Type any comment (e.g., "this is a terrible experience").

Select which Website_ID it belongs to (e.g., Site_A).

Click "Analyze & Add to Session".

You will see the sentiment analysis for that single comment.

This new comment is now temporarily added to the data for this session.

Step 2: Calculate Full Site WSI

Select a Website_ID from the dropdown.

Click "Calculate WSI".

The app will analyze all comments from the CSV for that site, plus any new comments you added from Step 1.

The WSI score, comment counts, and percentage breakdown will be displayed.

(Note: Added comments are only stored in memory. If you reload the page, they will be gone.)

Note on Embeddings

The included mock_embeddings.txt file is a tiny, 5-dimensional sample file used to make the project runnable without a large download.

For real-world accuracy, you should:

Download a pre-trained embedding model, such as GloVe (e.g., the 6B token, 100d file).

Update the EMBEDDING_FILE variable in sentiment_analyzer.py to point to your new file (e.g., glove.6B.100d.txt).

Update the file name in the fetch('mock_embeddings.txt') line in index.html to match.
