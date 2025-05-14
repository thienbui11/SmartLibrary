from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_book_embedding(book):
    """Trích xuất embedding từ tiêu đề và mô tả sách"""
    text = f"{book.title} {book.desc or ''}"
    embedding = model.encode(text)
    return embedding.astype(np.float32)
