
# ContextAware-Regulatory-GraphRAG

This repository contains the code and resources for the research paper on building a context-aware, multi-stage retrieval pipeline for question-answering on hierarchical regulatory documents. Our system, termed **GraphRAG**, leverages a knowledge graph to enhance context and a retrieve-and-re-rank architecture to achieve high-precision results on the **ObliQA-MP** dataset.

## 📋 Project Overview

Navigating complex regulatory and legal documents presents a significant challenge for traditional retrieval systems. The meaning of a specific passage is often dependent on its position within a deep structural hierarchy (e.g., sections, sub-sections, and guidance notes). Standard retrieval methods often fail to capture this structural context, leading to lower relevance scores.

This project addresses this challenge by:

1.  **Constructing a Knowledge Graph:** We model the entire corpus of regulatory documents as a knowledge graph, explicitly capturing hierarchical (`PARENT_OF`) and citation (`CITES`) relationships to preserve the documents' structural integrity.
    
2.  **Developing a Hybrid Retriever:** We combine a lexical **BM25** model with a dense **SentenceTransformer** model. The graph is used to create context-aware embeddings by concatenating text from neighboring nodes, which improves initial retrieval performance. The results are fused using **Reciprocal Rank Fusion (RRF)**.
    
3.  **Fine-Tuning a Cross-Encoder Re-ranker:** We fine-tune powerful cross-encoder models using a **hard-negative mining** strategy. This process uses BM25 to find lexically similar but semantically irrelevant passages, forcing the model to learn the fine-grained distinctions necessary for high-precision re-ranking.
    

The repository contains the full experimental pipeline, from graph construction to final evaluation, allowing for the complete reproduction of our results.

## 🛠️ Technical Details

### Knowledge Graph Construction

The foundation of our system is a multi-level knowledge graph constructed from a corpus of 40 regulatory documents. Using the `NetworkX` library, we parsed the JSON-formatted documents to create a directed graph containing nodes for **Documents**, **Passages**, **Named Entities**, and **Defined Terms**. Relationships between these nodes were established through several types of edges:

-   **Hierarchical Links:**  `PARENT_OF` edges were created based on the documents' decimal-based section numbering (e.g., section `1.2` is the parent of `1.2.1`), capturing the structural hierarchy of the regulations.
    
-   **Citation Links:**  `CITES` and `CITED_BY` edges were added based on an external cross-reference dataset, explicitly linking passages that refer to one another across the entire corpus.
    
-   **Containment Links:**  `CONTAINS` edges link documents to their constituent passages, while `MENTIONS` and `USES_TERM` edges link passages to the entities and terms they contain.
    

### Experimental Design

#### **Experiment 1: Baseline Retriever Evaluation**

The first experiment was designed to identify the most effective initial retrieval method. We systematically evaluated combinations of two key variables:

-   **Embedding Models:** We tested two pre-trained sentence-transformer models: `all-mpnet-base-v2` and `intfloat/e5-large-v2`.
    
-   **Context Strategies:** For each model, we generated four distinct sets of passage embeddings by varying the textual context: `passage_only`, `parent`, `parent_child`, and `full_neighborhood`.
    

For each of the 8 resulting embedding sets, we evaluated two retrieval methods against a manually curated ground truth (`qrels`) file using standard information retrieval metrics (`nDCG@10`, `MAP@10`, `Recall@10`):

-   **Dense Retrieval:** A pure semantic search using cosine similarity.
    
-   **Hybrid Retrieval:** A combination of dense retrieval and a lexical BM25 search, with the results merged using Reciprocal Rank Fusion (RRF).
    

#### **Experiment 2: Heuristic Graph Re-ranking**

This experiment aimed to measure the value of using simple, rule-based graph signals to re-rank the initial set of candidates. We conducted a comprehensive hyperparameter search, evaluating the impact of several heuristic bonuses including link-based bonuses (parent, sibling, cites), a centrality bonus (PageRank), and an isolation penalty.

#### **Experiment 3: Cross-Encoder Re-ranking**

The final experiment evaluates a more sophisticated re-ranking stage using fine-tuned cross-encoder models.

-   **Cross-Encoder Fine-Tuning:** To create high-quality re-rankers, we fine-tuned four different base models (`ms-marco-MiniLM-L-6-v2`, `all-mpnet-base-v2`, `ms-marco-TinyBERT-L-2-v2`, and `bert-base-uncased`). The training process was enhanced by hard negative mining using BM25 and validation-based early stopping to prevent overfitting.
    
-   **End-to-End Pipeline Evaluation:** The final pipeline consists of using the champion hybrid retriever from Experiment 1 to generate an initial set of 25 candidate passages, which are then re-ranked by each of the fine-tuned cross-encoders to identify the optimal system configuration.
    

## 📂 Repository Structure

```
.
├── 📁 Documents/
│   └── (Raw regulatory documents in JSON format)
├── 📁 QADataset/
│   ├── ObliQA_MultiPassage_train.json
│   ├── ObliQA_MultiPassage_val.json
│   └── ObliQA_MultiPassage_test.json
├── 📄 01_build_regulatory_knowledge_graph.ipynb
├── 📄 02_Generate_Embedding_Variants_MultiModel.ipynb
├── 📄 03_Fine_Tune_Cross_Encoders.ipynb
├── 📄 04_Experiment_1_Baseline_Retrieval.ipynb
├── 📄 05_Experiment_2_Graph_Reranking.ipynb
└── 📄 06_Experiment_3_Cross_Encoder_Pipeline.ipynb

```

## 🚀 Getting Started

### Prerequisites

This project is designed to be run in a **Google Colab** environment with a GPU.

### Setup

1.  **Access the Project Folder:** All necessary data, scripts, and notebooks are located in the following Google Drive folder. Please save a shortcut to your own Drive to access it from Colab.
    
    -   [**Project Google Drive Folder**](https://drive.google.com/drive/folders/1-vbEVai1LN2gVUKGR1zyhHrVw6TEsq7O?usp=sharing "null")
        
2.  **Mount Google Drive:** In each Colab notebook, you will need to mount your Google Drive to access the project files:
    
    ```
    from google.colab import drive
    drive.mount('/content/drive')
    
    ```
    

### Running the Experiments

The experiments are designed to be run sequentially, as each step generates artifacts required for the next. Please execute the notebooks in the following order:

1.  **`01_build_regulatory_knowledge_graph.ipynb`**:
    
    -   **Input:** Raw JSON documents.
        
    -   **Output:** The foundational `graph.gpickle` file.
        
2.  **`02_Generate_Embedding_Variants_MultiModel.ipynb`**:
    
    -   **Input:**  `graph.gpickle`.
        
    -   **Output:** Multiple sets of contextual passage embeddings.
        
3.  **`03_Fine_Tune_Cross_Encoders.ipynb`**:
    
    -   **Input:**  `ObliQA_MultiPassage_train.json`, `ObliQA_MultiPassage_val.json`.
        
    -   **Output:** Fine-tuned cross-encoder models saved to the `fine_tuned_cross_encoders` directory.
        
4.  **`04_Experiment_1_Baseline_Retrieval.ipynb`**:
    
    -   **Input:** Embeddings, `graph.gpickle`, `ObliQA_MultiPassage_test.json`.
        
    -   **Output:** A CSV of baseline retriever performance metrics.
        
5.  **`05_Experiment_2_Graph_Reranking.ipynb`**:
    
    -   **Input:** The best retriever configuration from Experiment 1.
        
    -   **Output:** A CSV evaluating heuristic-based graph re-ranking.
        
6.  **`06_Experiment_3_Cross_Encoder_Pipeline.ipynb`**:
    
    -   **Input:** The best retriever configuration and the fine-tuned cross-encoders.
        
    -   **Output:** A final CSV with the end-to-end pipeline performance.
        

## 📈 Results

Our experiments demonstrate the effectiveness of the multi-stage, context-aware pipeline. The best-performing configuration, combining the **e5-large hybrid retriever** with a **fine-tuned MiniLM cross-encoder**, significantly outperforms all baseline models.

_(This section can be updated with the final results tables )_

## 🎓 Citation

If you use this work, please cite our paper:

```
@article{YourLastName2025ContextAware,
  title   = {Context-Aware GraphRAG for High-Precision Question Answering on Hierarchical Regulatory Documents},
  author  = {Your Name, et al.},
  journal = {Proceedings of the Association for Computational Linguistics},
  year    = {2025}
}

```
