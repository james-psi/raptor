"""
RAPTOR Tree Retriever
Implements the retrieval process for finding relevant information in the tree structure.

Process Overview:
1. Configure retrieval parameters
2. Process query/question
3. Navigate tree to find relevant information
4. Return context for answer generation
"""

import logging
import os
from typing import Dict, List, Optional, Set, Tuple, Union

import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .EmbeddingModels import BaseEmbeddingModel, OpenAIEmbeddingModel
from .Retrievers import BaseRetriever
from .tree_structures import Node, Tree
from .utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances,
                    reverse_mapping)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class TreeRetrieverConfig:
    """
    Configuration for tree retrieval process.
    
    Step 1: Configuration Setup
    - Sets similarity thresholds
    - Configures traversal parameters
    - Sets up embedding model for queries
    """
    def __init__(
        self,
        tokenizer=None,
        threshold=None,
        top_k=None,
        selection_mode=None,
        context_embedding_model=None,
        embedding_model=None,
        num_layers=None,
        start_layer=None,
    ):
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer

        if threshold is None:
            threshold = 0.5
        if not isinstance(threshold, float) or not (0 <= threshold <= 1):
            raise ValueError("threshold must be a float between 0 and 1")
        self.threshold = threshold

        if top_k is None:
            top_k = 5
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be an integer and at least 1")
        self.top_k = top_k

        if selection_mode is None:
            selection_mode = "top_k"
        if not isinstance(selection_mode, str) or selection_mode not in [
            "top_k",
            "threshold",
        ]:
            raise ValueError(
                "selection_mode must be a string and either 'top_k' or 'threshold'"
            )
        self.selection_mode = selection_mode

        if context_embedding_model is None:
            context_embedding_model = "OpenAI"
        if not isinstance(context_embedding_model, str):
            raise ValueError("context_embedding_model must be a string")
        self.context_embedding_model = context_embedding_model

        if embedding_model is None:
            embedding_model = OpenAIEmbeddingModel()
        if not isinstance(embedding_model, BaseEmbeddingModel):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel"
            )
        self.embedding_model = embedding_model

        if num_layers is not None:
            if not isinstance(num_layers, int) or num_layers < 0:
                raise ValueError("num_layers must be an integer and at least 0")
        self.num_layers = num_layers

        if start_layer is not None:
            if not isinstance(start_layer, int) or start_layer < 0:
                raise ValueError("start_layer must be an integer and at least 0")
        self.start_layer = start_layer

    def log_config(self):
        config_log = """
        TreeRetrieverConfig:
            Tokenizer: {tokenizer}
            Threshold: {threshold}
            Top K: {top_k}
            Selection Mode: {selection_mode}
            Context Embedding Model: {context_embedding_model}
            Embedding Model: {embedding_model}
            Num Layers: {num_layers}
            Start Layer: {start_layer}
        """.format(
            tokenizer=self.tokenizer,
            threshold=self.threshold,
            top_k=self.top_k,
            selection_mode=self.selection_mode,
            context_embedding_model=self.context_embedding_model,
            embedding_model=self.embedding_model,
            num_layers=self.num_layers,
            start_layer=self.start_layer,
        )
        return config_log


class TreeRetriever(BaseRetriever):
    """
    Handles retrieval of relevant information from the tree.
    
    Process Flow:
    1. Initialize with tree and config
    2. Process input query
    3. Traverse tree to find relevant nodes
    4. Collect and return context
    """

    def __init__(self, config, tree) -> None:
        """
        Step 1: Initialize Retriever
        - Set up configurations
        - Store tree reference
        - Prepare embedding model
        """
        if not isinstance(tree, Tree):
            raise ValueError("tree must be an instance of Tree")

        if config.num_layers is not None and config.num_layers > tree.num_layers + 1:
            raise ValueError(
                "num_layers in config must be less than or equal to tree.num_layers + 1"
            )

        if config.start_layer is not None and config.start_layer > tree.num_layers:
            raise ValueError(
                "start_layer in config must be less than or equal to tree.num_layers"
            )

        self.tree = tree
        self.num_layers = (
            config.num_layers if config.num_layers is not None else tree.num_layers + 1
        )
        self.start_layer = (
            config.start_layer if config.start_layer is not None else tree.num_layers
        )

        if self.num_layers > self.start_layer + 1:
            raise ValueError("num_layers must be less than or equal to start_layer + 1")

        self.tokenizer = config.tokenizer
        self.top_k = config.top_k
        self.threshold = config.threshold
        self.selection_mode = config.selection_mode
        self.embedding_model = config.embedding_model
        self.context_embedding_model = config.context_embedding_model

        self.tree_node_index_to_layer = reverse_mapping(self.tree.layer_to_nodes)

        logging.info(
            f"Successfully initialized TreeRetriever with Config {config.log_config()}"
        )

    def create_embedding(self, text: str) -> List[float]:
        """
        Generates embeddings for the given text using the specified embedding model.

        Args:
            text (str): The text for which to generate embeddings.

        Returns:
            List[float]: The generated embeddings.
        """
        return self.embedding_model.create_embedding(text)

    def retrieve_information_collapse_tree(self, query: str, top_k: int, max_tokens: int) -> str:
        """
        Retrieves the most relevant information from the tree based on the query.

        Args:
            query (str): The query text.
            max_tokens (int): The maximum number of tokens.

        Returns:
            str: The context created using the most relevant nodes.
        """

        query_embedding = self.create_embedding(query)

        selected_nodes = []

        node_list = get_node_list(self.tree.all_nodes)

        embeddings = get_embeddings(node_list, self.context_embedding_model)

        distances = distances_from_embeddings(query_embedding, embeddings)

        indices = indices_of_nearest_neighbors_from_distances(distances)

        total_tokens = 0
        for idx in indices[:top_k]:

            node = node_list[idx]
            node_tokens = len(self.tokenizer.encode(node.text))

            if total_tokens + node_tokens > max_tokens:
                break

            selected_nodes.append(node)
            total_tokens += node_tokens

        context = get_text(selected_nodes)
        return selected_nodes, context

    def retrieve_information(
        self, current_nodes: List[Node], query: str, num_layers: int
    ) -> str:
        """
        Retrieves the most relevant information from the tree based on the query.

        Args:
            current_nodes (List[Node]): A List of the current nodes.
            query (str): The query text.
            num_layers (int): The number of layers to traverse.

        Returns:
            str: The context created using the most relevant nodes.
        """

        query_embedding = self.create_embedding(query)

        selected_nodes = []

        node_list = current_nodes

        for layer in range(num_layers):

            embeddings = get_embeddings(node_list, self.context_embedding_model)

            distances = distances_from_embeddings(query_embedding, embeddings)

            indices = indices_of_nearest_neighbors_from_distances(distances)

            if self.selection_mode == "threshold":
                best_indices = [
                    index for index in indices if distances[index] > self.threshold
                ]

            elif self.selection_mode == "top_k":
                best_indices = indices[: self.top_k]

            nodes_to_add = [node_list[idx] for idx in best_indices]

            selected_nodes.extend(nodes_to_add)

            if layer != num_layers - 1:

                child_nodes = []

                for index in best_indices:
                    child_nodes.extend(node_list[index].children)

                # take the unique values
                child_nodes = list(dict.fromkeys(child_nodes))
                node_list = [self.tree.all_nodes[i] for i in child_nodes]

        context = get_text(selected_nodes)
        return selected_nodes, context

    def get_relevant_nodes(
        self,
        query_embedding: List[float],
        list_nodes: List[Node],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        selection_mode: Optional[str] = None,
    ) -> List[Node]:
        """
        Step 2: Find Relevant Nodes
        
        Process:
        1. Calculate similarity between query and nodes
        2. Select nodes based on criteria:
           - Top-K most similar
           - Above threshold
        3. Return selected nodes
        """
        if top_k is None:
            top_k = self.top_k
        if threshold is None:
            threshold = self.threshold
        if selection_mode is None:
            selection_mode = self.selection_mode

        embeddings = get_embeddings(list_nodes, self.context_embedding_model)
        distances = distances_from_embeddings(query_embedding, embeddings)
        indices = indices_of_nearest_neighbors_from_distances(distances)

        if selection_mode == "threshold":
            best_indices = [
                index for index in indices if distances[index] > threshold
            ]
        elif selection_mode == "top_k":
            best_indices = indices[:top_k]

        nodes_to_add = [list_nodes[idx] for idx in best_indices]
        return nodes_to_add

    def retrieve(
        self,
        query: str,
        start_layer: Optional[int] = None,
        num_layers: Optional[int] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = True,
    ) -> Union[str, Tuple[str, Dict[int, List[Node]]]]:
        """
        Step 3: Tree Traversal and Retrieval
        
        Process:
        1. Create query embedding
        2. Start at specified layer
        3. For each layer:
           - Find similar nodes
           - Get children of similar nodes
           - Move to next layer
        4. Collect all relevant information
        5. Return context (and optionally layer info)

        Args:
            query: The query string to search for
            start_layer: Which layer to start searching from (defaults to top)
            num_layers: How many layers to traverse down
            top_k: How many top matches to consider
            max_tokens: Maximum number of tokens to include in context
            collapse_tree: Whether to collapse all layers into one context
            return_layer_information: Whether to return layer-wise node information
        """
        if start_layer is None:
            start_layer = self.start_layer or self.tree.num_layers
        if num_layers is None:
            num_layers = self.num_layers or self.tree.num_layers
        if top_k is None:
            top_k = self.top_k

        # Create query embedding
        query_embedding = self.embedding_model.create_embedding(query)
        
        # Track nodes at each layer
        layer_to_nodes = {}
        current_layer = start_layer
        
        # Get initial nodes from starting layer
        if current_layer in self.tree.layer_to_nodes:
            current_nodes = self.tree.layer_to_nodes[current_layer]
        else:
            current_nodes = []

        # Traverse layers
        for layer in range(current_layer, max(current_layer - num_layers, -1), -1):
            relevant_nodes = self.get_relevant_nodes(
                query_embedding, current_nodes, top_k=top_k
            )
            layer_to_nodes[layer] = relevant_nodes
            
            if layer > 0:
                # Get children for next layer
                children_sets = get_children(relevant_nodes)
                # Flatten the sets of children into a single list of unique indices
                children = list(set().union(*children_sets))
                current_nodes = [
                    self.tree.all_nodes[child_idx]
                    for child_idx in children
                ]

        # Collect and format context
        if collapse_tree:
            all_nodes = [node for layer in layer_to_nodes.values() for node in layer]
            # Filter nodes based on max_tokens if specified
            if max_tokens:
                selected_nodes = []
                total_tokens = 0
                for node in all_nodes:
                    node_tokens = len(self.tokenizer.encode(node.text))
                    if total_tokens + node_tokens > max_tokens:
                        break
                    selected_nodes.append(node)
                    total_tokens += node_tokens
                context = get_text(selected_nodes)
            else:
                context = get_text(all_nodes)
        else:
            context = get_text(layer_to_nodes[min(layer_to_nodes.keys())])

        if return_layer_information:
            return context, layer_to_nodes
        return context
