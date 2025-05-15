# Smart Library System

## Description

The **Smart Library System** is an intelligent platform designed to assist users in managing, searching, and retrieving book-related information using Artificial Intelligence. The system integrates technologies such as OCR (Optical Character Recognition), face recognition, and a chatbot powered by Retrieval-Augmented Generation (RAG) to enhance the user experience in a library setting. It allows users to add books via ISBN scanning or cover image, extract introductions from books, query book content via chatbot, and even log in through facial recognition. This platform supports librarians, students, and readers in accessing and organizing knowledge efficiently.

## Model Fine-Tuned

- **Embedding Model for Chatbot**: Sentence Transformers (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
- **OCR Model**: VietOCR (`vgg_transformer`) for Vietnamese book title recognition
- **Face Recognition**: Dlib-based or InsightFace-based model for facial login

## Features

- **OCR Book Title & Author Recognition**: Extracts titles and author names from scanned book covers using Vietnamese OCR.
- **Book Introduction Extraction**: Automatically extracts the preface/introduction from scanned PDFs using text detection + OCR.
- **Chatbot with Book Knowledge**: Chat with the system to get summarized information or answers based on book introductions.
- **Face Recognition Login**: Users can authenticate via facial recognition, improving convenience and security.
- **ISBN Scanning Support**: Retrieves metadata via ISBN from external APIs.
- **QR Code Scanning**: Alternate login or checkout method using QR codes.

## Data Collection & Processing

1. **OCR Pipeline**:
   - Uses PaddleOCR + VietOCR to extract text from images of book covers.
   - Preprocessing includes noise removal, contrast adjustment, and font thickening for accuracy.

2. **PDF Processing**:
   - Extracts text from scanned books using layout-aware OCR.
   - Saves introduction sections into ChromaDB with embedding vectors.

3. **Face Recognition Data**:
   - User face images are stored as encodings in a vector store to support fast face matching.

## Model Optimization

1. **OCR Optimization**:
   - Combined PaddleOCR detection + VietOCR recognition for higher Vietnamese OCR accuracy.
   - Fine-tuned detection threshold and line merging postprocessing.

2. **Embedding & Search**:
   - Sentence embeddings indexed in ChromaDB for fast semantic search on book introductions.
   - Supports RAG (Retrieval-Augmented Generation) for chatbot queries.

3. **Face Recognition**:
   - Used Dlib or InsightFace to embed user face images and compare using cosine similarity.
   - Optimized lighting and angle normalization for webcam input.

## Deployment

1. **Backend with Flask**:
   - Flask serves REST API endpoints for chatbot, OCR, face login, and book management.

2. **Frontend**:
   - Web interface using HTML/CSS/JS (or optionally Django) with pages for scanning, login, and chatbot queries.

3. **Deployment**:
   - Backend containerized for easy deployment on local machines or servers.
   - Compatible with Nvidia Docker for GPU acceleration in OCR/LLM tasks.

## Technologies Used

- **LangChain**: Manages the flow for chatbot RAG queries.
- **ChromaDB**: Stores book introductions as embeddings and enables vector search.
- **PaddleOCR + VietOCR**: Vietnamese OCR pipeline.
- **Flask**: Python web backend serving APIs.
- **SQL Server**: Stores structured book and user data.
- **Docker**: Deployment and environment consistency.
- **face_recognition (Dlib)**: For user login via webcam face scan.
- **Llama.cpp**: Lightweight LLM for answering book-related queries.

## Future Enhancements

- **LLM Fine-Tuning for Book QA**: Fine-tune a model on Vietnamese book datasets to improve answer quality.
- **Mobile App Integration**: Extend to Android/iOS for barcode scanning and chatbot access.
- **Admin Dashboard**: Add librarian dashboard to track book additions and user activity.
- **Book Recommendation Engine**: Recommend books based on user interests and chat history.
