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
from raptor import ClusterTreeConfig, RetrievalAugmentationConfig

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
    """Visualize the actual tree structure with proper hierarchical layout and cluster grouping."""
    plt.figure(figsize=(20, 12), facecolor='white')
    G = nx.DiGraph()  # Use directed graph for hierarchical structure
    
    # Track nodes by level for positioning
    nodes_by_level = {}
    
    # Track clusters for positioning
    clusters_by_level = {}
    
    # First pass: Add all nodes to the graph and find root nodes
    root_nodes = []
    for layer, nodes in tree.layer_to_nodes.items():
        if layer not in nodes_by_level:
            nodes_by_level[layer] = []
            clusters_by_level[layer] = {}
        
        # Group nodes by their parent to identify clusters
        node_parents = {}
        for node in nodes:
            node_id = str(node.index)
            display_text = node.text[:50] + '...' if len(node.text) > 50 else node.text
            G.add_node(node_id, text=display_text, level=layer)
            nodes_by_level[layer].append(node_id)
            
            # Find parent of this node
            parent_id = None
            for other_node in tree.all_nodes.values():
                if hasattr(other_node, 'children') and node.index in other_node.children:
                    parent_id = str(other_node.index)
                    break
            node_parents[node_id] = parent_id
            
            # If no parent found, it's a root node
            if parent_id is None:
                root_nodes.append(node_id)
        
        # Group nodes by their parent (cluster)
        for node_id, parent_id in node_parents.items():
            if parent_id not in clusters_by_level[layer]:
                clusters_by_level[layer][parent_id] = []
            clusters_by_level[layer][parent_id].append(node_id)
    
    # Print information about root nodes
    print(f"\nFound {len(root_nodes)} root node(s):")
    for root_id in root_nodes:
        root_text = G.nodes[root_id]['text']
        print(f"Root node {root_id}: {root_text[:100]}...")
    
    # Second pass: Add edges
    for node_id, node in tree.all_nodes.items():
        if hasattr(node, 'children') and node.children:
            for child_id in node.children:
                G.add_edge(str(node_id), str(child_id))
    
    # Calculate positions using cluster-aware layout
    pos = {}
    num_layers = len(nodes_by_level)
    
    # Position nodes layer by layer with cluster awareness
    for layer in range(num_layers):
        clusters = clusters_by_level[layer]
        total_clusters = len(clusters)
        cluster_idx = 0
        
        for parent_id, cluster_nodes in clusters.items():
            num_nodes = len(cluster_nodes)
            cluster_center = (cluster_idx - (total_clusters - 1) / 2) / max(1, total_clusters - 1)
            
            # Position nodes within cluster
            for i, node in enumerate(cluster_nodes):
                # x-coordinate: cluster center + offset within cluster
                node_offset = (i - (num_nodes - 1) / 2) / (max(1, num_nodes) * 2)
                x = cluster_center + node_offset
                
                # y-coordinate: layer based
                y = layer / max(1, num_layers - 1)
                
                pos[node] = np.array([x, y])
            
            cluster_idx += 1
    
    # Draw edges first (behind nodes)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=20, width=1.5, arrowstyle='->')
    
    # Draw nodes layer by layer with different colors
    colors = ['#FFB6C1', '#98FB98', '#87CEEB', '#DDA0DD', '#F0E68C']  # Pastel colors for each layer
    
    # Draw cluster backgrounds first
    for layer, clusters in clusters_by_level.items():
        for parent_id, cluster_nodes in clusters.items():
            if len(cluster_nodes) > 1:  # Only draw background for actual clusters
                # Get cluster node positions
                cluster_pos = np.array([pos[node] for node in cluster_nodes])
                # Calculate cluster boundary
                min_x = np.min(cluster_pos[:, 0]) - 0.1
                max_x = np.max(cluster_pos[:, 0]) + 0.1
                min_y = np.min(cluster_pos[:, 1]) - 0.05
                max_y = np.max(cluster_pos[:, 1]) + 0.05
                
                # Draw cluster background
                plt.fill_between(
                    [min_x, max_x],
                    [min_y, min_y],
                    [max_y, max_y],
                    color=colors[layer % len(colors)],
                    alpha=0.2
                )
    
    # Then draw nodes
    for layer, nodes in nodes_by_level.items():
        # Special handling for root nodes
        node_colors = []
        node_sizes = []
        for node in nodes:
            if node in root_nodes:
                node_colors.append('#FF0000')  # Red for root nodes
                node_sizes.append(4000)  # Larger size for root nodes
            else:
                node_colors.append(colors[layer % len(colors)])
                node_sizes.append(3000)
        
        nx.draw_networkx_nodes(G, pos, 
                             nodelist=nodes,
                             node_color=node_colors,
                             node_size=node_sizes,
                             alpha=0.7)
    
    # Add labels with smaller font and wrapped text
    labels = {}
    for node in G.nodes():
        text = G.nodes[node]['text']
        layer = G.nodes[node]['level']
        # Special label for root nodes
        if node in root_nodes:
            prefix = f"ROOT NODE (Layer {layer})\n"
        else:
            prefix = f"Layer {layer}\n"
        # Wrap text for better display
        wrapped_text = prefix + '\n'.join(
            [text[i:i+20] for i in range(0, len(text), 20)][:3]
        )
        labels[node] = wrapped_text
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title("RAPTOR Tree Structure\nHierarchical View with Cluster Information\n(Root nodes in red, Clusters shown as backgrounds)",
             pad=20, fontsize=14, fontweight='bold')
    
    # Remove axes
    plt.axis('off')
    
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
    
    # Initialize RAPTOR with custom configuration
    tree_config = ClusterTreeConfig(
        num_layers=3,  # Increase number of layers
        reduction_dimension=3,  # Reduce this to allow more layers
        max_tokens=100,  # Keep chunk size reasonable
        summarization_length=150  # Slightly longer summaries
    )
    config = RetrievalAugmentationConfig(tree_builder_config=tree_config)
    
    # Initialize RAPTOR with the custom config
    RA = RetrievalAugmentation(config=config)
    
    # Step 2: Build and visualize the tree
    print("\nStep 2: Building RAPTOR tree...")
    RA.add_documents(text)
    
    # Print tree structure information before visualization
    print("\nTree Structure Information:")
    print(f"Number of layers: {len(RA.tree.layer_to_nodes)}")
    for layer, nodes in RA.tree.layer_to_nodes.items():
        print(f"Layer {layer}: {len(nodes)} nodes")
    
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