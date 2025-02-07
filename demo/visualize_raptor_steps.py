import matplotlib.pyplot as plt
import networkx as nx
import os
import sys
from pathlib import Path
import numpy as np

# Add the parent directory to Python path to find the raptor module
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from raptor import RetrievalAugmentation, RetrievalAugmentationConfig

def visualize_tokenization(text, max_tokens=100, filename='1_tokenization'):
    """Visualize how the text is split into chunks."""
    plt.figure(figsize=(15, 8), facecolor='white')
    
    # Create a grid of boxes representing chunks
    num_chunks = len(text) // max_tokens + 1
    rows = int(np.sqrt(num_chunks))
    cols = (num_chunks + rows - 1) // rows
    
    for i in range(num_chunks):
        row = i // cols
        col = i % cols
        chunk = text[i*max_tokens:(i+1)*max_tokens]
        plt.text(col, -row, f"Chunk {i+1}\n{chunk[:50]}...",
                bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.5'),
                ha='center', va='center', fontsize=8)
    
    plt.xlim(-0.5, cols-0.5)
    plt.ylim(-rows+0.5, 0.5)
    plt.axis('off')
    plt.title("Step 1: Text Tokenization", pad=20, fontsize=14, fontweight='bold')
    
    # Save the visualization
    output_dir = current_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{filename}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"Tokenization visualization saved to: {output_path}")

def visualize_embeddings(num_chunks=35, filename='2_embeddings'):
    """Visualize the embedding process."""
    plt.figure(figsize=(15, 8), facecolor='white')
    
    # Create a visualization showing text chunks being converted to vectors
    for i in range(num_chunks):
        # Draw text box
        plt.text(0.2, 1-i/num_chunks, f"Chunk {i+1}",
                bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.5'),
                ha='center', va='center', fontsize=8)
        
        # Draw arrow
        plt.arrow(0.3, 1-i/num_chunks, 0.2, 0, head_width=0.02,
                 head_length=0.02, fc='gray', ec='gray')
        
        # Draw vector representation
        plt.text(0.6, 1-i/num_chunks, f"[0.1, 0.8, -0.3, ...]",
                bbox=dict(facecolor='lightgreen', edgecolor='black', boxstyle='round,pad=0.5'),
                ha='center', va='center', fontsize=8)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title("Step 2: Embedding Generation using text-embedding-3-small",
             pad=20, fontsize=14, fontweight='bold')
    
    output_path = current_dir / 'visualizations' / f"{filename}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"Embedding visualization saved to: {output_path}")

def visualize_clustering(filename='3_clustering'):
    """Visualize the hierarchical clustering process."""
    plt.figure(figsize=(15, 8), facecolor='white')
    
    # Create a simple dendrogram-like structure
    G = nx.Graph()
    
    # Add nodes for chunks and clusters
    chunks = [f"Chunk {i+1}" for i in range(6)]  # Simplified example
    clusters = ["Cluster 1", "Cluster 2"]
    root = "Root Cluster"
    
    # Add edges to form the hierarchy
    G.add_edges_from([
        (clusters[0], chunks[0]),
        (clusters[0], chunks[1]),
        (clusters[0], chunks[2]),
        (clusters[1], chunks[3]),
        (clusters[1], chunks[4]),
        (clusters[1], chunks[5]),
        (root, clusters[0]),
        (root, clusters[1])
    ])
    
    # Create a hierarchical layout
    pos = nx.spring_layout(G)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, nodelist=chunks,
                          node_color='lightblue', node_size=2000)
    nx.draw_networkx_nodes(G, pos, nodelist=clusters,
                          node_color='lightgreen', node_size=2000)
    nx.draw_networkx_nodes(G, pos, nodelist=[root],
                          node_color='lightcoral', node_size=2000)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title("Step 3: Hierarchical Clustering", pad=20, fontsize=14, fontweight='bold')
    
    output_path = current_dir / 'visualizations' / f"{filename}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"Clustering visualization saved to: {output_path}")

def visualize_summarization(filename='4_summarization'):
    """Visualize the recursive summarization process."""
    plt.figure(figsize=(15, 8), facecolor='white')
    
    # Create example text and summaries
    texts = {
        'cluster1': "Once upon a time, there was a kind girl named Cinderella...",
        'cluster2': "The prince held a grand ball to find his bride...",
        'summary1': "Introduction to Cinderella's character and situation",
        'summary2': "The prince's ball and search for bride",
        'final': "A fairy tale about Cinderella finding love at a royal ball"
    }
    
    # Create a hierarchical visualization
    levels = 3
    y_positions = np.linspace(0.8, 0.2, levels)
    
    # Draw original texts
    plt.text(0.25, y_positions[0], texts['cluster1'][:50] + '...',
            bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.5'),
            ha='center', va='center', fontsize=8, wrap=True)
    plt.text(0.75, y_positions[0], texts['cluster2'][:50] + '...',
            bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.5'),
            ha='center', va='center', fontsize=8, wrap=True)
    
    # Draw first level summaries
    plt.text(0.25, y_positions[1], texts['summary1'],
            bbox=dict(facecolor='lightgreen', edgecolor='black', boxstyle='round,pad=0.5'),
            ha='center', va='center', fontsize=8)
    plt.text(0.75, y_positions[1], texts['summary2'],
            bbox=dict(facecolor='lightgreen', edgecolor='black', boxstyle='round,pad=0.5'),
            ha='center', va='center', fontsize=8)
    
    # Draw final summary
    plt.text(0.5, y_positions[2], texts['final'],
            bbox=dict(facecolor='lightcoral', edgecolor='black', boxstyle='round,pad=0.5'),
            ha='center', va='center', fontsize=8)
    
    # Draw arrows
    plt.arrow(0.25, y_positions[0]-0.05, 0, -0.15, head_width=0.02,
             head_length=0.02, fc='gray', ec='gray')
    plt.arrow(0.75, y_positions[0]-0.05, 0, -0.15, head_width=0.02,
             head_length=0.02, fc='gray', ec='gray')
    plt.arrow(0.25, y_positions[1]-0.05, 0.15, -0.15, head_width=0.02,
             head_length=0.02, fc='gray', ec='gray')
    plt.arrow(0.75, y_positions[1]-0.05, -0.15, -0.15, head_width=0.02,
             head_length=0.02, fc='gray', ec='gray')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title("Step 4: Recursive Summarization using GPT-4 Turbo",
             pad=20, fontsize=14, fontweight='bold')
    
    output_path = current_dir / 'visualizations' / f"{filename}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"Summarization visualization saved to: {output_path}")

def visualize_tree(filename='5_tree'):
    """Visualize the final tree structure."""
    plt.figure(figsize=(15, 10), facecolor='white')
    
    G = nx.Graph()
    
    # Create a tree structure
    root = "Final Summary"
    clusters = ["Beginning", "Middle", "End"]
    leaves = [
        "Introduction", "Family Setup",  # Beginning
        "Ball Preparation", "At the Ball",  # Middle
        "Shoe Search", "Happy Ending"   # End
    ]
    
    # Add edges
    G.add_edges_from([
        (root, clusters[0]),
        (root, clusters[1]),
        (root, clusters[2]),
        (clusters[0], leaves[0]),
        (clusters[0], leaves[1]),
        (clusters[1], leaves[2]),
        (clusters[1], leaves[3]),
        (clusters[2], leaves[4]),
        (clusters[2], leaves[5])
    ])
    
    # Use a hierarchical layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw the tree
    nx.draw_networkx_nodes(G, pos, nodelist=[root],
                          node_color='lightcoral', node_size=3000)
    nx.draw_networkx_nodes(G, pos, nodelist=clusters,
                          node_color='lightgreen', node_size=2500)
    nx.draw_networkx_nodes(G, pos, nodelist=leaves,
                          node_color='lightblue', node_size=2000)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title("Step 5: Final Tree Structure", pad=20, fontsize=14, fontweight='bold')
    
    output_path = current_dir / 'visualizations' / f"{filename}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"Tree visualization saved to: {output_path}")

def visualize_retrieval(question="How did Cinderella reach her happy ending?", filename='6_retrieval'):
    """Visualize the retrieval process."""
    plt.figure(figsize=(15, 8), facecolor='white')
    
    # Draw the question at the top
    plt.text(0.5, 0.9, f"Question: {question}",
            bbox=dict(facecolor='yellow', edgecolor='black', boxstyle='round,pad=0.5'),
            ha='center', va='center', fontsize=10)
    
    # Draw relevant passages
    passages = [
        "Cinderella's kindness and perseverance...",
        "The magical help from the hazel tree...",
        "The prince recognized her as his true bride...",
        "They lived happily ever after..."
    ]
    
    for i, passage in enumerate(passages):
        plt.text(0.5, 0.7 - i*0.15, passage,
                bbox=dict(facecolor='lightgreen', edgecolor='black', boxstyle='round,pad=0.5'),
                ha='center', va='center', fontsize=8)
    
    # Draw the final answer at the bottom
    answer = "Cinderella reached her happy ending through her kindness, magical help, and true love..."
    plt.text(0.5, 0.1, f"Answer: {answer}",
            bbox=dict(facecolor='lightcoral', edgecolor='black', boxstyle='round,pad=0.5'),
            ha='center', va='center', fontsize=10)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title("Step 6: Tree-based Retrieval and Answer Generation",
             pad=20, fontsize=14, fontweight='bold')
    
    output_path = current_dir / 'visualizations' / f"{filename}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"Retrieval visualization saved to: {output_path}")

def visualize_all_steps():
    """Create visualizations for all steps of RAPTOR processing."""
    print("\nCreating step-by-step visualizations of RAPTOR processing...")
    
    # Read the sample text
    with open('demo/sample.txt', 'r') as file:
        text = file.read()
    
    # Create visualizations for each step
    visualize_tokenization(text)
    visualize_embeddings()
    visualize_clustering()
    visualize_summarization()
    visualize_tree()
    visualize_retrieval()
    
    print("\nDone! Check the visualizations folder for all step-by-step diagrams.")

if __name__ == '__main__':
    visualize_all_steps() 