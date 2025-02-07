# RAPTOR Process Documentation

## Overview
This document details the step-by-step process of how RAPTOR (Retrieval Augmented Processing Tree for Organized Responses) works, from initial text processing to final retrieval.

## 1. Initial Setup
When creating a new RAPTOR instance (`RA = RetrievalAugmentation()`):
- Initializes configurations for:
  - Tree building
  - Retrieval parameters
  - QA model settings
- Sets up ClusterTreeBuilder (default)
- No tree exists at this point

## 2. Text Processing
When adding documents (`RA.add_documents(text)`):
- Takes input text
- Uses tiktoken tokenizer
- Splits text into manageable chunks
  - Default: ~100 tokens per chunk
  - Maintains semantic coherence where possible

## 3. Leaf Node Creation
For each text chunk:
- Creates a new node with:
  - Unique index
  - Original text chunk
  - Embeddings (using configured model, default: OpenAI)
  - Empty children set
- These become the bottom layer (Layer 0) of the tree

## 4. Tree Construction (Bottom-up)
### Layer 0 (Bottom Layer)
- Contains all leaf nodes
- Holds original text chunks
- Full embeddings for each chunk

### Building Upper Layers
For each layer (default: 5 layers):
1. Takes nodes from layer below
2. Groups similar nodes using RAPTOR_Clustering:
   - Reduces embedding dimensionality
   - Applies clustering algorithm
   - Groups semantically related content
3. For each cluster:
   - Combines text from all nodes
   - Creates summary of combined text
   - Generates embeddings for summary
   - Creates new parent node containing:
     - Summary text
     - Links to child nodes
     - Summary embeddings
4. Process continues until:
   - Reaches specified layer count
   - Or too few nodes to cluster further

## 5. Tree Finalization
Creates final Tree object containing:
- Complete node dictionary (index → node)
- Root nodes (top layer)
- Leaf nodes (bottom layer)
- Layer-to-nodes mapping
- Layer count

## 6. Retriever Setup
- Creates TreeRetriever instance
- Configures:
  - Similarity thresholds
  - Top-k settings
  - Layer traversal parameters

## 7. Retrieval Process
When answering questions:
1. Question Processing:
   - Takes question text
   - Creates question embeddings

2. Tree Traversal:
   - Starts at specified layer (usually top)
   - For each layer:
     - Finds most similar nodes
     - Explores promising children
     - Moves down the tree

3. Context Collection:
   - Gathers most relevant nodes
   - Combines their text

4. Answer Generation:
   - Sends question + context to QA model
   - Returns generated answer

## Notes
- Tree construction is optimized for the entire document set
- Each layer provides increasingly summarized content
- Retrieval efficiency comes from top-down traversal
- Current version creates new tree for new documents
- Tree structure enables efficient information retrieval 