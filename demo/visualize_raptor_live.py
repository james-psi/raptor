import os
import sys
from pathlib import Path

# Add the parent directory to Python path to find the raptor module
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Setup Azure OpenAI environment variables BEFORE importing raptor modules
os.environ["AZURE_OPENAI_API_KEY"] = "KEY"
os.environ["AZURE_OPENAI_ENDPOINT"] = "ENDPOINT"

# Now import other required packages
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import logging
import tiktoken

# Configure logging to show processing steps
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# Now we can safely import raptor modules
from raptor import RetrievalAugmentation
from raptor.tree_structures import Node, Tree

def save_visualization(plt, name):
    """Helper function to save visualizations."""
    output_dir = Path(__file__).resolve().parent / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{name}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"\nVisualization saved to: {output_path}")

def get_text_chunks(text, max_tokens=100):
    """Split text into chunks using RAPTOR's tokenization."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for token in tokens:
        if current_length >= max_tokens:
            chunks.append(tokenizer.decode(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(token)
        current_length += 1
    
    if current_chunk:
        chunks.append(tokenizer.decode(current_chunk))
    
    return chunks

def visualize_text_chunks(text, max_tokens=100):
    """Visualize how the text is split into chunks."""
    chunks = get_text_chunks(text, max_tokens)
    
    plt.figure(figsize=(15, 8), facecolor='white')
    
    # Calculate grid layout
    rows = int(np.sqrt(len(chunks)))
    cols = (len(chunks) + rows - 1) // rows
    
    for i, chunk in enumerate(chunks):
        row = i // cols
        col = i % cols
        # Show first 50 characters of each chunk
        display_text = chunk[:50] + '...' if len(chunk) > 50 else chunk
        plt.text(col, -row, f"Chunk {i+1}\n{display_text}",
                bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.5'),
                ha='center', va='center', fontsize=8)
    
    plt.xlim(-0.5, cols-0.5)
    plt.ylim(-rows+0.5, 0.5)
    plt.axis('off')
    plt.title("Step 1: Text Chunking (max 100 tokens per chunk)", 
             pad=20, fontsize=14, fontweight='bold')
    
    save_visualization(plt, "1_text_chunks")
    return chunks

def visualize_tree_structure(tree):
    """Visualize the actual tree structure."""
    plt.figure(figsize=(15, 10), facecolor='white')
    G = nx.Graph()
    
    def add_node_to_graph(node, parent_id=None, level=0):
        node_id = str(node.index)  # Use node's index as identifier
        # Use actual text from the node, truncated for display
        display_text = node.text[:50] + '...' if node.text else 'Empty Node'
        G.add_node(node_id, text=display_text, level=level)
        
        if parent_id:
            G.add_edge(parent_id, node_id)
        
        # Add child nodes using the children indices
        for child_idx in node.children:
            child_node = tree.all_nodes[child_idx]
            add_node_to_graph(child_node, node_id, level+1)
    
    # Start with root nodes
    for root_node in tree.root_nodes:
        add_node_to_graph(tree.all_nodes[root_node])
        
    # Create layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Get nodes by level
    nodes_by_level = {}
    for node, attr in G.nodes(data=True):
        level = attr['level']
        if level not in nodes_by_level:
            nodes_by_level[level] = []
        nodes_by_level[level].append(node)
    
    # Draw nodes level by level with different colors
    colors = ['lightcoral', 'lightgreen', 'lightblue']
    for level, nodes in nodes_by_level.items():
        nx.draw_networkx_nodes(G, pos, nodelist=nodes,
                             node_color=colors[level % len(colors)],
                             node_size=2000)
    
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos,
                          labels={node: G.nodes[node]['text'][:30] + '...' 
                                 for node in G.nodes()},
                          font_size=6)
    
    plt.title("Step 2: RAPTOR Tree Construction",
             pad=20, fontsize=14, fontweight='bold')
    
    save_visualization(plt, "2_tree_structure")

def visualize_retrieval_process(question, context_nodes, answer):
    """Visualize the actual retrieval and answer generation."""
    plt.figure(figsize=(15, 8), facecolor='white')
    
    # Draw the question
    plt.text(0.5, 0.9, f"Question: {question}",
            bbox=dict(facecolor='yellow', edgecolor='black', boxstyle='round,pad=0.5'),
            ha='center', va='center', fontsize=10)
    
    # Draw retrieved contexts
    for i, node in enumerate(context_nodes):
        text = node.text[:50] + '...' if node.text else 'Empty Node'
        plt.text(0.5, 0.7 - i*0.15, text,
                bbox=dict(facecolor='lightgreen', edgecolor='black', boxstyle='round,pad=0.5'),
                ha='center', va='center', fontsize=8)
    
    # Draw the answer
    plt.text(0.5, 0.1, f"Answer: {answer}",
            bbox=dict(facecolor='lightcoral', edgecolor='black', boxstyle='round,pad=0.5'),
            ha='center', va='center', fontsize=10)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title("Step 3: Question Answering",
             pad=20, fontsize=14, fontweight='bold')
    
    save_visualization(plt, "3_qa_process")

def main():
    print("\nStarting RAPTOR visualization with Cinderella story...")
    
    # Read the story
    with open('demo/sample.txt', 'r') as file:
        text = file.read()
    print("\nLoaded text:", text[:100], "...")
    
    # Step 1: Visualize text chunking
    print("\nStep 1: Chunking text...")
    chunks = visualize_text_chunks(text)
    print(f"Created {len(chunks)} chunks")
    
    # Initialize RAPTOR
    RA = RetrievalAugmentation()
    
    # Step 2: Build and visualize the tree
    print("\nStep 2: Building RAPTOR tree...")
    RA.add_documents(text)
    visualize_tree_structure(RA.tree)
    
    # Step 3: Question answering
    question = "How did Cinderella reach her happy ending?"
    print(f"\nStep 3: Processing question: {question}")
    
    answer = RA.answer_question(question=question)
    print("\nAnswer:", answer)
    
    # Visualize the retrieval process
    print("\nVisualizing retrieval process...")
    # Use all_nodes attribute and convert dictionary values to list
    context_nodes = list(RA.tree.all_nodes.values())[:4]
    visualize_retrieval_process(question, context_nodes, answer)
    
    print("\nVisualization complete! Check the visualizations folder for all steps:")
    print("1. Text chunking (1_text_chunks.png)")
    print("2. Tree construction (2_tree_structure.png)")
    print("3. Question answering (3_qa_process.png)")

if __name__ == '__main__':
    main() 