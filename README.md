# ATS Vector RAG Project

This project implements a vector-based Retrieval-Augmented Generation (RAG)
pipeline for an Applicant Tracking System (ATS).

## Structure
- data/raw: raw datasets (CSV)
- data/processed: intermediate artifacts (JSONL)
- src: processing scripts

## Steps
1. Build atomic CV–JD evaluation cases
2. Extract entities (skills, education, certifications)
3. Generate embeddings and vector store
4. RAG-based evidence retrieval
