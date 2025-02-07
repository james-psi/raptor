"""
RAPTOR Cluster Tree Builder
Implements tree construction using clustering-based approach for organizing nodes.

Process Overview:
1. Configure clustering parameters
2. Process input text into nodes
3. Build tree using clustering algorithm
"""

import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Set

from .cluster_utils import ClusteringAlgorithm, RAPTOR_Clustering
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_structures import Node, Tree
from .utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances, split_text)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class ClusterTreeConfig(TreeBuilderConfig):
    """
    Configuration for cluster-based tree building.
    
    Step 1: Configuration Setup
    - Sets clustering parameters
    - Configures dimension reduction
    - Inherits base tree building settings
    """
    def __init__(
        self,
        reduction_dimension=10,
        clustering_algorithm=RAPTOR_Clustering,  # Default to RAPTOR clustering
        clustering_params={},  # Pass additional params as a dict
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reduction_dimension = reduction_dimension
        self.clustering_algorithm = clustering_algorithm
        self.clustering_params = clustering_params

    def log_config(self):
        base_summary = super().log_config()
        cluster_tree_summary = f"""
        Reduction Dimension: {self.reduction_dimension}
        Clustering Algorithm: {self.clustering_algorithm.__name__}
        Clustering Parameters: {self.clustering_params}
        """
        return base_summary + cluster_tree_summary


class ClusterTreeBuilder(TreeBuilder):
    """
    Builds tree structure using clustering approach.
    
    Process Flow:
    1. Initialize with configuration
    2. Process input text
    3. Build tree using clustering
    4. Create final tree structure
    """

    def __init__(self, config) -> None:
        """
        Step 1: Initialize Builder
        - Set up configurations
        - Prepare clustering algorithm
        - Initialize base tree builder
        """
        super().__init__(config)

        if not isinstance(config, ClusterTreeConfig):
            raise ValueError("config must be an instance of ClusterTreeConfig")
        self.reduction_dimension = config.reduction_dimension
        self.clustering_algorithm = config.clustering_algorithm
        self.clustering_params = config.clustering_params

        logging.info(
            f"Successfully initialized ClusterTreeBuilder with Config {config.log_config()}"
        )

    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = True,
    ) -> Dict[int, Node]:
        """
        Step 2: Tree Construction
        
        Process:
        1. Start with bottom layer nodes
        2. For each layer:
           a. Get current layer nodes
           b. Apply clustering algorithm
           c. Create parent nodes for clusters
           d. Build connections between layers
        3. Continue until reaching top layer
        
        Parameters:
        - current_level_nodes: Nodes in current layer
        - all_tree_nodes: All nodes in tree
        - layer_to_nodes: Mapping of layers to nodes
        - use_multithreading: Whether to use parallel processing
        """
        logging.info("Using Cluster TreeBuilder")

        next_node_index = len(all_tree_nodes)

        def process_cluster(
            cluster, new_level_nodes, next_node_index, summarization_length, lock
        ):
            """
            Step 3: Cluster Processing
            
            For each cluster:
            1. Get texts from cluster nodes
            2. Create summary of combined texts
            3. Generate embeddings for summary
            4. Create new parent node
            5. Update tree structure
            """
            try:
                # Get and validate node texts
                node_texts = get_text(cluster)
                if not isinstance(node_texts, str):
                    logging.error(f"Invalid node_texts type: {type(node_texts)}")
                    node_texts = str(node_texts)  # Convert to string if not already
                
                if not node_texts.strip():
                    logging.error("Empty node texts")
                    node_texts = "Empty text"  # Provide default text
                
                # Get and validate summarized text
                try:
                    summarized_text = self.summarize(
                        context=node_texts,
                        max_tokens=summarization_length,
                    )
                except Exception as e:
                    logging.error(f"Error in summarization: {str(e)}")
                    summarized_text = node_texts[:summarization_length]  # Use truncated original text as fallback
                
                if not isinstance(summarized_text, str):
                    logging.error(f"Invalid summarized_text type: {type(summarized_text)}")
                    summarized_text = str(summarized_text)  # Convert to string if not already
                    
                if not summarized_text.strip():
                    logging.error("Empty summarized text")
                    summarized_text = node_texts[:summarization_length]  # Use truncated original text as fallback

                # Log lengths safely
                try:
                    node_text_length = len(self.tokenizer.encode(node_texts))
                    summary_length = len(self.tokenizer.encode(summarized_text))
                    logging.info(
                        f"Node Texts Length: {node_text_length}, Summarized Text Length: {summary_length}"
                    )
                except Exception as e:
                    logging.error(f"Error encoding text: {str(e)}")

                # Create new node
                __, new_parent_node = self.create_node(
                    next_node_index, summarized_text, {node.index for node in cluster}
                )

                with lock:
                    new_level_nodes[next_node_index] = new_parent_node
                    
            except Exception as e:
                logging.error(f"Error in process_cluster: {str(e)}")
                # Create a fallback node with safe values
                fallback_text = "Error processing cluster"
                __, new_parent_node = self.create_node(
                    next_node_index, fallback_text, {node.index for node in cluster}
                )
                with lock:
                    new_level_nodes[next_node_index] = new_parent_node

            return True  # Indicate successful processing

        # Step 4: Layer Construction
        for layer in range(self.num_layers):
            new_level_nodes = {}
            logging.info(f"Constructing Layer {layer}")

            node_list_current_layer = get_node_list(current_level_nodes)

            # Check if we can create more layers
            if len(node_list_current_layer) <= self.reduction_dimension + 1:
                self.num_layers = layer
                logging.info(
                    f"Stopping Layer construction: Cannot Create More Layers. Total Layers in tree: {layer}"
                )
                break

            # Apply clustering
            clusters = self.clustering_algorithm.perform_clustering(
                node_list_current_layer,
                self.cluster_embedding_model,
                reduction_dimension=self.reduction_dimension,
                **self.clustering_params,
            )

            # Process clusters (parallel or sequential)
            lock = Lock()
            summarization_length = self.summarization_length
            logging.info(f"Summarization Length: {summarization_length}")

            if use_multithreading:
                with ThreadPoolExecutor() as executor:
                    for cluster in clusters:
                        executor.submit(
                            process_cluster,
                            cluster,
                            new_level_nodes,
                            next_node_index,
                            summarization_length,
                            lock,
                        )
                        next_node_index += 1
                    executor.shutdown(wait=True)
            else:
                for cluster in clusters:
                    process_cluster(
                        cluster,
                        new_level_nodes,
                        next_node_index,
                        summarization_length,
                        lock,
                    )
                    next_node_index += 1

            # Update tree structure
            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

            # Create tree object for current state
            tree = Tree(
                all_tree_nodes,
                layer_to_nodes[layer + 1],
                layer_to_nodes[0],
                layer + 1,
                layer_to_nodes,
            )

        return current_level_nodes
