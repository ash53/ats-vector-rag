# ATS Vector RAG (Phase 1)

This repository contains the Phase 1 implementation of a **Vector-based Retrieval-Augmented Generation (RAG)** pipeline for an Applicant Tracking System (ATS).

The goal of this phase is to **compare vector-based retrieval with graph-based retrieval (GraphRAG)** before selecting an approach for deeper analysis.

---

## Dataset
We use the following public dataset:
- AI Recruitment Pipeline Dataset (Kaggle)  
  https://www.kaggle.com/datasets/yaswanthkumary/ai-recruitment-pipeline-dataset/data

The dataset contains resumes, job descriptions, hiring decisions, and human-written decision rationales.

---

## Pipeline Overview

1. **Case Construction**
   - Normalize each recruitment record into a canonical CV–JD–Decision case

2. **Entity Extraction**
   - Extract skills, degrees, and certifications from resumes and job descriptions

3. **Vector Indexing**
   - Build multi-view embeddings (context + skill alignment)
   - Index using FAISS

4. **Retrieval & Evidence Construction**
   - Retrieve similar historical cases
   - Stratify by decision (select/reject)
   - Construct explainable evidence chunks

---

## Project Structure

```text
src/
  step1_cases.py
  step2_entities.py
  step3_embeddings_faiss.py
  step4_retrieval.py

data/vocab/
  skills.txt
  certifications.txt
