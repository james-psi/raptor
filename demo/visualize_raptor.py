import matplotlib.pyplot as plt
import networkx as nx
import os
import sys
from pathlib import Path
import shutil

# Add the parent directory to Python path to find the raptor module
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from raptor import RetrievalAugmentation
from raptor.tree_structures import Node, Tree

def check_graphviz_installation():
    """Check if Graphviz is installed and accessible."""
    if not shutil.which('dot'):
        print("\nGraphviz is not installed or not in PATH. Please follow these steps:")
        print("\n1. Install Graphviz:")
        if sys.platform == "win32":
            print("   - Download from: https://graphviz.org/download/")
            print("   - During installation, make sure to select 'Add to PATH'")
            print("   - After installation, restart your terminal")
        elif sys.platform == "darwin":
            print("   - Run: brew install graphviz")
        else:
            print("   - Run: sudo apt-get install graphviz")
        print("\n2. Verify installation:")
        print("   - Open a new terminal and run: dot -V")
        return False
    return True

def create_mermaid_diagram(filename='raptor_mermaid'):
    """Create a Mermaid.js compatible diagram as fallback."""
    mermaid_code = """
graph TB
    A[Input Document] --> B[Tokenization]
    B --> C[Embedding Generation]
    C --> D[Hierarchical Clustering]
    D --> E[Recursive Summarization]
    E --> F[Tree Construction]
    F --> G[Tree-based Retrieval]
    G --> H[Question Answering]
    """
    
    output_dir = current_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{filename}.md"
    
    with open(output_path, 'w') as f:
        f.write("```mermaid\n")
        f.write(mermaid_code)
        f.write("\n```")
    
    print(f"\nCreated Mermaid diagram at {output_path}")
    print("You can view this diagram by:")
    print("1. Opening in VS Code with Mermaid extension")
    print("2. Pasting into a Markdown viewer that supports Mermaid")
    print("3. Using the Mermaid Live Editor: https://mermaid.live")

def create_tree_visualization(tree, filename='raptor_tree'):
    """Create a tree visualization using matplotlib."""
    if not hasattr(tree, 'root'):
        print("\nNo valid tree structure found.")
        return
        
    plt.figure(figsize=(15, 10))
    G = nx.Graph()
    
    def add_node_to_graph(node, parent_id=None):
        # Create a shortened version of the text for display
        display_text = node.text[:30] + '...' if node.text and len(node.text) > 30 else 'Empty Node'
        node_id = str(id(node))
        
        # Add node
        G.add_node(node_id, text=display_text)
        
        # Add edge to parent
        if parent_id:
            G.add_edge(parent_id, node_id)
        
        # Process children
        if node.children:
            for child in node.children:
                add_node_to_graph(child, node_id)
    
    # Build the graph
    add_node_to_graph(tree.root)
    
    # Create layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, labels={node: G.nodes[node]['text'] for node in G.nodes()},
            node_color='lightgreen', node_size=2000, font_size=6,
            font_weight='bold', node_shape='s',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    
    # Save the plot
    output_dir = current_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{filename}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Tree visualization saved to {output_path}")

def create_architecture_diagram(filename='raptor_architecture'):
    """Create a diagram showing both the architecture and data flow."""
    plt.figure(figsize=(15, 10), facecolor='white')
    
    # Define the main pipeline steps
    steps = [
        ('Input\nDocument', 'Cinderella story\nfrom sample.txt'),
        ('Tokenization', '35 text chunks\nMax tokens: 100'),
        ('Embedding\nGeneration', 'text-embedding-3-small\nmodel'),
        ('Hierarchical\nClustering', 'RAPTOR_Clustering\nReduction Dim: 10'),
        ('Recursive\nSummarization', 'GPT-4 Turbo\nSummary length: 100'),
        ('Tree\nConstruction', '1 layer tree with\nsummarized nodes'),
        ('Tree-based\nRetrieval', 'Top K: 5\nThreshold: 0.5'),
        ('Question\nAnswering', 'GPT-4 Turbo\nQA Model')
    ]
    
    # Create a vertical layout with slight offset for details
    main_positions = {}
    detail_positions = {}
    for i, (step, detail) in enumerate(steps):
        y_pos = 1 - i/7  # Spread across vertical space
        main_positions[step] = (0.3, y_pos)  # Main steps on the left
        detail_positions[detail] = (0.7, y_pos)  # Details on the right
    
    # Create the graph
    G = nx.DiGraph()
    
    # Add main pipeline nodes and detail nodes
    for (step, detail) in steps:
        G.add_node(step, type='main')
        G.add_node(detail, type='detail')
        G.add_edge(step, detail, style='dotted')
        if i < len(steps)-1:
            G.add_edge(step, steps[i+1][0])
    
    # Draw the main pipeline
    pos = {**main_positions, **detail_positions}
    
    # Draw main nodes
    main_nodes = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'main']
    nx.draw_networkx_nodes(G, pos, nodelist=main_nodes, 
                          node_color='lightblue', node_size=3000,
                          node_shape='s')
    
    # Draw detail nodes
    detail_nodes = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'detail']
    nx.draw_networkx_nodes(G, pos, nodelist=detail_nodes,
                          node_color='lightgreen', node_size=3000,
                          node_shape='s')
    
    # Draw edges
    edges = [(u, v) for (u, v, d) in G.edges(data=True) if d.get('style') != 'dotted']
    dotted_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d.get('style') == 'dotted']
    
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='gray', arrows=True)
    nx.draw_networkx_edges(G, pos, edgelist=dotted_edges, edge_color='gray', 
                          style='dotted', arrows=False)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    # Add title
    plt.title("RAPTOR Processing Pipeline for Cinderella Story", 
             pad=20, fontsize=14, fontweight='bold')
    
    # Save the visualization
    output_dir = current_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{filename}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"\nArchitecture diagram saved to: {output_path}")
    print(f"You can open this image with:")
    print(f"- Windows: double-click the file")
    print(f"- Mac: open {output_path}")
    print(f"- Linux: xdg-open {output_path}")

def visualize_example():
    """Create visualizations showing how RAPTOR processed the Cinderella story."""
    print("\nCreating RAPTOR visualization...")
    create_architecture_diagram()
    print("\nDone! Check the visualizations folder for the output.")

if __name__ == '__main__':
    visualize_example() 