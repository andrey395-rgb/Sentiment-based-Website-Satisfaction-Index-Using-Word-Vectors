# Sentiment-Based Website Satisfaction Index (GUI Application)

## 1. Project Overview

This project implements a **sentiment analysis system** that classifies user feedback into **positive**, **neutral**, and **negative** categories. It uses the mathematical framework of **vector spaces** and **inner products** to quantify user sentiment.

By representing textual data (user comments) as numerical vectors, the model applies **cosine similarity** to measure how closely a comment aligns with predefined sentiment directions. These results are then aggregated into a single metric called the **Website Satisfaction Index (WSI)**, providing a clear, data-driven measure of overall user satisfaction.

This repository includes a **Python GUI application** built using **Tkinter** for interactive analysis and management.
---

## 2. Core Concept

The sentiment of each user comment is determined using the following steps:

1. **Vectorization**
   Each word in a comment is converted into a high-dimensional vector using a pre-trained embedding model (e.g., *GloVe*).

2. **Document Vector**
   The vectors of all words in a comment are averaged to create a single **document vector** `d` representing the comment’s overall meaning.

3. **Reference Vectors**
   Two reference sentiment vectors are defined:

   * **Positive Vector (s_pos)** – average of seed words like *good*, *excellent*, *amazing*
   * **Negative Vector (s_neg)** – average of seed words like *bad*, *terrible*, *awful*

4. **Cosine Similarity**
   The model computes the cosine similarity between `d` and both reference vectors.

5. **Classification**
   Based on directional closeness:

   * If `d` is closer to `s_pos` → **Positive**
   * If `d` is closer to `s_neg` → **Negative**
   * If `d` lies between the two within a `NEUTRALITY_THRESHOLD` → **Neutral**

---

## 3. File Structure

```
/
├── sentiment_gui.py        # MAIN APP: GUI frontend (run this file)
├── sentiment_model.py      # BACKEND: Core logic for sentiment and WSI
│
├── comments.csv            # Data file storing website comments
├── mock_embeddings.txt     # Sample embedding file (5D vectors for testing)
├── batch_comments.txt      # Sample batch input file for demo purposes
│
├── README.md               # Project documentation (this file)
│
└── (Deprecated Files)
    ├── sentiment_analyzer.py   # Old command-line version
    └── index.html              # Old web-based version
```

---

## 4. How to Run the Application

### Step 1: Install Dependencies

Install the required Python libraries:

```bash
pip install pandas numpy tkinterdnd2
```

> Note: `tkinter` is included with most Python installations by default.

### Step 2: Run the Application

Launch the GUI by running:

```bash
python sentiment_gui.py
```

The application will load the embedding model, then open the main window automatically.

---

## 5. Application Features

The GUI is divided into **four main tabs**, each with a specific purpose:

### 🟦 Tab 1: Analyze Website

**Function:** Computes and displays the complete WSI analysis for a chosen website.

**How to Use:**

1. Select a **Website_ID** from the dropdown menu.
2. Click **"Calculate WSI"**.
3. The results will display:

   * WSI score
   * Sentiment counts (positive, neutral, negative)
   * Full list of labeled comments
4. A **“Save As…”** dialog will open, allowing you to export the summary as a `.txt` file (e.g., `WSI_Summary_Site_A.txt`).

---

### 🟩 Tab 2: Add New Site

**Function:** Adds a new, blank website entry to `comments.csv`.

**How to Use:**

1. Enter the name of the new site (e.g., `Site_C`).
2. Click **"Add New Website"**.
3. The site will now appear in all dropdown menus.

---

### 🟨 Tab 3: Add Single Comment

**Function:** Adds a single comment to a selected website.

**How to Use:**

1. Choose a **Website_ID**.
2. Type your comment in the text box.
3. Click **"Add Comment to CSV"** to permanently save it.

---

### 🟧 Tab 4: Batch Upload

**Function:** Uploads multiple comments from a `.txt` file at once.

**How to Use:**

1. Select the **Website_ID** you want to add comments to.
2. Choose your file via:

   * **Drag and Drop:** Drop your `.txt` file (e.g., `batch_comments.txt`) onto the drop zone.
   * **Browse:** Click **"Browse..."** to select a file manually.
3. Click **"Submit Batch File"** to upload all comments.

---

## 6. Notes on Embeddings

The included `mock_embeddings.txt` is a small, 5-dimensional embedding file meant for demonstration.
For real-world accuracy and better sentiment differentiation:

1. Download a **pre-trained embedding model**, such as **GloVe 6B (100d)**.
   Example source: [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)

2. Update the following line in `sentiment_model.py`:

   ```python
   EMBEDDING_FILE = 'glove.6B.100d.txt'
   ```

3. Restart the application after updating the file path.

---

## 7. Future Improvements

* Integration with real-time web feedback systems
* Advanced vector representations using transformer models (e.g., BERT)
* Interactive dashboards for trend visualization
* Database integration for large-scale deployment

---

## 8. License

This project is released for academic and educational purposes.
All contributors retain rights to their respective code and documentation.

---

Would you like me to format this for GitHub (e.g., add markdown tables, collapsible sections, or visual badges like “Built with Tkinter”)?
