# LLM Projects: RAG & Search-Augmented Answer Engine

This repository contains two Python projects exploring how Large Language Models (LLMs) can retrieve relevant information and generate useful, context-aware answers.

## Projects

### 1. RAG Document Q&A

**Technologies:** Python, OpenAI API, Sentence Transformers, ChromaDB

An end-to-end Retrieval-Augmented Generation (RAG) application that answers questions using information retrieved from documents.

**Key features**

* Document ingestion and text chunking.
* Text embeddings using Sentence Transformers.
* Vector storage and semantic search using ChromaDB.
* Context-grounded answer generation using the OpenAI API.

**How it works**

1. Documents are ingested and split into smaller chunks.
2. Embeddings are generated for the document chunks.
3. ChromaDB stores the embeddings for semantic retrieval.
4. Relevant passages are retrieved based on the user's question.
5. The LLM generates an answer grounded in the retrieved context.

**Source code:** [`rag/`](./rag)

### 2. Search-Augmented Answer Engine

**Technologies:** Python, Google Serper API, OpenAI API

A conversational research assistant that combines live web search with LLM-based answer generation. Instead of relying only on the model's existing knowledge, it retrieves search results to provide relevant, source-informed responses.

**Key features**

* Retrieves organic Google search results using the Serper API.
* Uses the OpenAI `gpt-4.1-mini` model to synthesize answers from search results.
* Uses structured prompts to encourage concise, relevant answers.

**How it works**

1. The user submits a question.
2. The application retrieves web search results.
3. Search results are passed to the LLM as supporting context.
4. The model generates an answer based on the retrieved information and its instructions.
5. Conversation history is retained for follow-up questions.

**Source code:** [`PerplexityDemo.py`](./PerplexityDemo.py)

## Technology Stack

* **Programming language:** Python
* **LLM integration:** OpenAI API
* **Retrieval and search:** Sentence Transformers, ChromaDB, Google Serper API

## Getting Started

### Prerequisites

* Python installed on your system.
* An OpenAI API key.
* A Google Serper API key for the search-augmented application.


## Learning Objectives

These projects explore practical LLM application development, including:

* Retrieval-Augmented Generation (RAG).
* Embeddings and semantic search.
* Vector databases and information retrieval.
* Search-augmented answer generation.
* Prompt engineering and conversational context management.
* Integrating external APIs with LLM applications.

## Author

**Pooja Yeli**

[GitHub Profile](https://github.com/PoojaYeli)
