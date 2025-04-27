
import os
import faiss
import pandas as pd
import numpy as np
import gradio as gr
import faiss
import pickle
from sklearn.metrics import ndcg_score
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string
import re



# 1. Load FAISS indices
sbert_index = faiss.read_index("sbert_index.faiss")
sbert_fine_tuned = faiss.read_index("sbert_fine_tuned_index.faiss")
tfidf_index = faiss.read_index("tfidf_index.faiss")

# 2. Load metadata dictionary
with open("vector_db.pkl", "rb") as f:
    vector_db = pickle.load(f)


# Load SBERT models
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
sbert_model_finetuned = SentenceTransformer("/finetuned_sbert/fine_tuned_sbert_model")

# Load TFIDF Vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)


df = pd.read_csv("test_df.csv")

# Download required resources (only once)
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def preprocess_text(
    text,
    lowercase=True,
    remove_punct=False,
    remove_stopwords=False,
    lemmatize=False,
    stem=False,
    normalize_whitespace=True,
    remove_invisibles=True
):
    # Remove invisible characters like \u200b (zero-width space)
    if remove_invisibles:
        text = text.replace('\u200b', '')

    # Normalize newlines and tabs to spaces
    text = re.sub(r'[\r\n\t]', ' ', text)

    # Normalize whitespace
    if normalize_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()

    # Lowercase
    if lowercase:
        text = text.lower()

    # Remove punctuation
    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    tokens = text.split()

    # Stopwords removal
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words]

    # Lemmatization or stemming
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    elif stem:
        tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)



evaluation_results = {
    "SBERT": {
        "Average Precision@5": 0.7992,
        "Average Recall@5": 0.7992,
        "Average F1@5": 0.7992,
        "MRR": 0.9092,
        "NDCG@5": 0.8966
    },
    "TFIDF": {
        "Average Precision@5": 0.7162,
        "Average Recall@5": 0.7162,
        "Average F1@5": 0.7162,
        "MRR": 0.8709,
        "NDCG@5": 0.8666
    },
    "SBERT Fine-Tuned": {
        "Average Precision@5": 0.8738,
        "Average Recall@5": 0.8738,
        "Average F1@5": 0.8738,
        "MRR": 0.9502,
        "NDCG@5": 0.9280
    }
}

# --- 3. Helper Functions ---

def retrieve_similar(post_id_or_text, retrieval_technique, is_custom_query=False):
    if retrieval_technique == "SBERT":
        mode = "sbert"
        index = sbert_index
        model = sbert_model
    elif retrieval_technique == "SBERT Fine-Tuned":
        mode = "sbert_finetuned"
        index = sbert_fine_tuned
        model = sbert_model_finetuned
    else:  # TFIDF
        mode = "tfidf"
        index = tfidf_index
        model = None  # TFIDF uses vectorizer, not a model

    if is_custom_query:
        preprocessed_text = preprocess_text(
            post_id_or_text,
            lowercase=True,
            remove_punct=(mode == "tfidf"),
            remove_stopwords=(mode == "tfidf"),
            lemmatize=(mode == "tfidf"),
            stem=False,
            normalize_whitespace=True,
            remove_invisibles=True
        )

        if model is None:  # TFIDF
            query_vector = tfidf_vectorizer.transform([preprocessed_text]).toarray().astype(np.float32)
        else:  # SBERT or Fine-Tuned
            query_vector = model.encode([preprocessed_text])
            query_vector = np.asarray(query_vector, dtype=np.float32)

        faiss.normalize_L2(query_vector)
        ground_truth = set()  # no ground truth for custom text
    else:
        query_vector = np.expand_dims(vector_db[post_id_or_text][f'embedding_{mode}'], axis=0)
        ground_truth = set(vector_db[post_id_or_text]["ground_truth"])

    adaptive_k = len(ground_truth) if not is_custom_query else 5
    D, I = index.search(query_vector, adaptive_k)
    all_ids = list(vector_db.keys())
    top_ids = [all_ids[i] for i in I[0] if ( all_ids[i] != post_id_or_text)][:adaptive_k]

    cards_html = ""
    for pid in top_ids:
        cards_html += f"""<div class='recommend-card'>{vector_db[pid]['post']}<div class='icons'>
        <span>👍</span><span>💬</span><span>🔁</span></div></div>"""

    if  not ground_truth and not is_custom_query:
        metrics_html = "<div class='metrics-card'>⚠️ No ground truth available for custom text.</div>"
        return cards_html, metrics_html

    tp = len(set(top_ids) & ground_truth) 
    precision = tp / adaptive_k if not is_custom_query else 0
    recall = tp / len(ground_truth) if not is_custom_query else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mrr = next((1 / (rank + 1) for rank, pid in enumerate(top_ids) if pid in ground_truth), 0) if not is_custom_query else 0
    relevance = [1 if pid in ground_truth else 0 for pid in top_ids]

    if len(relevance) > 1:
        ndcg = ndcg_score([relevance], [list(range(len(relevance), 0, -1))])
    else:
        ndcg = 0

    metrics_html = "<div class='metrics-card'><h4>📈 Retrieval Metrics</h4><ul>"
    for k, v in {
        f"Precision@{adaptive_k}": precision ,
        f"Recall@{adaptive_k}": recall,
        "F1 Score": f1,
        "MRR": mrr,
        f"NDCG@{adaptive_k}": ndcg,
    }.items():
        metrics_html += f"<li><b>{k}:</b> {v:.4f}</li>"
    metrics_html += "</ul></div>"

    return cards_html, metrics_html


def show_evaluation_results(retrieval_technique):
    metrics = evaluation_results.get(retrieval_technique, {})
    if not metrics:
        return "<div>No evaluation results available.</div>"

    html = "<div class='metrics-card'><h4>📊 Overall Retrieval performance for the current algorithm</h4><ul>"
    for k, v in metrics.items():
        html += f"<li><b>{k}:</b> {v:.4f}</li>"
    html += "</ul></div>"
    return html

def sample_posts():
    sampled = df.sample(5).reset_index(drop=True)
    post_texts = []
    post_ids = []
    for _, row in sampled.iterrows():
        post_texts.append(row["generated_paragraph"])
        post_ids.append(row["id"])
    return post_texts, post_ids

# --- 4. Gradio Interface ---

with gr.Blocks() as demo:
    gr.HTML("""<style>
    .scrollable-left {
      max-height: 80vh;
      overflow-y: auto;
      overflow-x: hidden;
      padding: 10px;
    }
    .post-card, .recommend-card {
      background: white;
      border: 1px solid #333;
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0px 2px 5px rgba(0,0,0,0.2);
      margin-bottom: 15px;
      font-size: 1rem;
      color: #000;
    }
    .icons {
      display: flex; gap: 15px;
      margin-top: 10px;
      color: #777;
      font-size: 0.9em;
    }
    .metrics-card {
      background: #f9f9f9;
      border: 1px solid #ccc;
      border-radius: 12px;
      padding: 16px;
      margin-top: 20px;
      font-size: 0.95rem;
    }
    </style>""")

    retrieval_method = gr.State("SBERT")
    sampled_post_ids = gr.State([])

    with gr.Row():
        technique_dropdown = gr.Dropdown(
            choices=["SBERT", "SBERT Fine-Tuned", "TFIDF"],
            value="SBERT",
            label="🔍 Select Retrieval Technique"
        )
    overall_metrics_html = gr.HTML()

    with gr.Row():
      custom_query = gr.Textbox(placeholder="🔍 Type any post to search...", label="Custom Query Text")
      custom_query_btn = gr.Button("🔎 Search with Custom Query")


    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📜 All Posts")
            refresh_btn = gr.Button("🔄 Sample 5 New Posts")
            with gr.Column(elem_classes="scrollable-left") as posts_column:
                post_blocks = []
                see_similar_buttons = []
                for _ in range(5):
                    with gr.Column() as wrapper:
                        html_block = gr.HTML()
                        button = gr.Button("🔍 See Similar Posts", elem_classes="see-similar-btn")
                        post_blocks.append(html_block)
                        see_similar_buttons.append(button)

        with gr.Column(scale=2):
            gr.Markdown("### 🔁 Recommended Posts")
            rec_html = gr.HTML("Click a post to see similar ones here")
            metrics_html = gr.HTML()

    custom_query_btn.click(
    fn=lambda query, method: retrieve_similar(query, method, is_custom_query=True),
    inputs=[custom_query, retrieval_method],
    outputs=[rec_html, metrics_html]
)
    technique_dropdown.change(
        fn=lambda x: (x, show_evaluation_results(x)),
        inputs=[technique_dropdown],
        outputs=[retrieval_method, overall_metrics_html]
    )

    def refresh_sample():
        claims, ids = sample_posts()
        post_htmls = []
        for claim in claims:
            html = f"""
            <div class="post-card">
              {claim}
              <div class="icons">
                <span>👍 Like</span>
                <span>💬 Comment</span>
                <span>🔁 Share</span>
              </div>
            </div>
            """
            post_htmls.append(html)
        return post_htmls + [ids]

    refresh_btn.click(
        fn=refresh_sample,
        inputs=[],
        outputs=post_blocks + [sampled_post_ids]
    )

    demo.load(
        fn=refresh_sample,
        inputs=[],
        outputs=post_blocks + [sampled_post_ids]
    )

    for i in range(5):
        see_similar_buttons[i].click(
            fn=lambda ids, idx, method: retrieve_similar(ids[idx], method),
            inputs=[sampled_post_ids, gr.State(i), retrieval_method],
            outputs=[rec_html, metrics_html]
        )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 Overall Evaluation Results")
            overall_metrics_html


if __name__ == "__main__":
    demo.launch()