# 🔍 SBERT and TFIDF Post Retrieval System

This is a Gradio-powered application that retrieves similar posts based on an input post or a custom query.  
It supports three different retrieval techniques:

- **SBERT (Sentence-BERT) Pretrained**
- **SBERT Fine-Tuned** on custom data
- **TFIDF (Term Frequency-Inverse Document Frequency)**

You can select any method and explore how each technique retrieves similar posts from the dataset!

---

## 🚀 How It Works

1. **Sample Posts**: The app randomly selects 5 posts from the dataset.
2. **View Recommendations**: Click "See Similar Posts" to retrieve related posts based on the selected method.
3. **Custom Query**: You can also input your own text to find similar posts dynamically.
4. **Performance Metrics**: Each retrieval shows live metrics like Precision@5, Recall@5, F1 Score, MRR, and NDCG.
5. **Global Evaluation**: Displays the overall performance of each retrieval method.

---

## 📚 Models and Indexes Used

- `all-MiniLM-L6-v2` from SentenceTransformers (Pretrained SBERT)
- Custom Fine-Tuned SBERT Model
- TFIDF Vectorizer trained on the dataset
- FAISS indexes for fast similarity search (cosine similarity)

---

## 🛠 Files and Components

| File/Folder                    | Purpose                           |
| :----------------------------- | :-------------------------------- |
| `app.py`                       | Main Gradio application           |
| `requirements.txt`             | Python dependencies               |
| `sbert_index.faiss`            | Pretrained SBERT embeddings index |
| `sbert_fine_tuned_index.faiss` | Fine-Tuned SBERT embeddings index |
| `tfidf_index.faiss`            | TFIDF vectors index               |
| `vector_db.pkl`                | Metadata about each post          |
| `tfidf_vectorizer.pkl`         | Trained TFIDF vectorizer          |
| `fine_tuned_sbert_model/`      | Fine-tuned SBERT model folder     |

---

## ⚡ Instructions to Use

1. Select a retrieval technique (SBERT, SBERT Fine-Tuned, or TFIDF).
2. Browse sampled posts and click on a post to find similar ones.
3. Alternatively, input your own custom text and search.
4. View retrieval results along with live evaluation metrics.
5. Compare methods to explore retrieval performance!

---

## 📈 Global Retrieval Evaluation (Summary)

| Retrieval Method | Average Precision@5 | Average Recall@5 | Average F1@5 | MRR    | NDCG@5 |
| :--------------- | :------------------ | :--------------- | :----------- | :----- | :----- |
| SBERT            | 0.7992              | 0.7992           | 0.7992       | 0.9092 | 0.8966 |
| TFIDF            | 0.7162              | 0.7162           | 0.7162       | 0.8709 | 0.8666 |
| SBERT Fine-Tuned | 0.8738              | 0.8738           | 0.8738       | 0.9502 | 0.9280 |

---

## 🙌 Credits

- Built with [Gradio](https://gradio.app/) and [Hugging Face Spaces](https://huggingface.co/spaces)
- Sentence embeddings powered by [Sentence Transformers](https://www.sbert.net/)
- FAISS used for efficient vector similarity search

---

## 📬 Contact

Feel free to reach out if you have questions or suggestions!

- **Author**: _[Taha Draoui]_
- **Email**: _[tahadr@umich.edu]_

---
