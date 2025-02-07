"""
RAPTOR (Retrieval Augmented Processing Tree for Organized Responses)
Main interface for document processing, tree construction, and question answering.

Step-by-Step Process:
1. Initialize RAPTOR with configurations
2. Add documents to create the tree
3. Use the tree for retrieving information and answering questions
"""

import logging
import pickle

from .cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from .EmbeddingModels import BaseEmbeddingModel
from .QAModels import BaseQAModel, GPT3TurboQAModel
from .SummarizationModels import BaseSummarizationModel
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_retriever import TreeRetriever, TreeRetrieverConfig
from .tree_structures import Node, Tree

# Define a dictionary to map supported tree builders to their respective configs
supported_tree_builders = {"cluster": (ClusterTreeBuilder, ClusterTreeConfig)}

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class RetrievalAugmentationConfig:
    """
    Configuration class for RAPTOR.
    
    Step 1: Configuration Setup
    - Defines all parameters for tree building
    - Sets up retrieval parameters
    - Configures QA model settings
    """
    def __init__(
        self,
        tree_builder_config=None,
        tree_retriever_config=None,  # Change from default instantiation
        qa_model=None,
        embedding_model=None,
        summarization_model=None,
        tree_builder_type="cluster",
        # New parameters for TreeRetrieverConfig and TreeBuilderConfig
        # TreeRetrieverConfig arguments
        tr_tokenizer=None,
        tr_threshold=0.5,
        tr_top_k=5,
        tr_selection_mode="top_k",
        tr_context_embedding_model="OpenAI",
        tr_embedding_model=None,
        tr_num_layers=None,
        tr_start_layer=None,
        # TreeBuilderConfig arguments
        tb_tokenizer=None,
        tb_max_tokens=100,
        tb_num_layers=5,
        tb_threshold=0.5,
        tb_top_k=5,
        tb_selection_mode="top_k",
        tb_summarization_length=100,
        tb_summarization_model=None,
        tb_embedding_models=None,
        tb_cluster_embedding_model="OpenAI",
    ):
        # Validate tree_builder_type
        if tree_builder_type not in supported_tree_builders:
            raise ValueError(
                f"tree_builder_type must be one of {list(supported_tree_builders.keys())}"
            )

        # Validate qa_model
        if qa_model is not None and not isinstance(qa_model, BaseQAModel):
            raise ValueError("qa_model must be an instance of BaseQAModel")

        if embedding_model is not None and not isinstance(
            embedding_model, BaseEmbeddingModel
        ):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel"
            )
        elif embedding_model is not None:
            if tb_embedding_models is not None:
                raise ValueError(
                    "Only one of 'tb_embedding_models' or 'embedding_model' should be provided, not both."
                )
            tb_embedding_models = {"EMB": embedding_model}
            tr_embedding_model = embedding_model
            tb_cluster_embedding_model = "EMB"
            tr_context_embedding_model = "EMB"

        if summarization_model is not None and not isinstance(
            summarization_model, BaseSummarizationModel
        ):
            raise ValueError(
                "summarization_model must be an instance of BaseSummarizationModel"
            )

        elif summarization_model is not None:
            if tb_summarization_model is not None:
                raise ValueError(
                    "Only one of 'tb_summarization_model' or 'summarization_model' should be provided, not both."
                )
            tb_summarization_model = summarization_model

        # Set TreeBuilderConfig
        tree_builder_class, tree_builder_config_class = supported_tree_builders[
            tree_builder_type
        ]
        if tree_builder_config is None:
            tree_builder_config = tree_builder_config_class(
                tokenizer=tb_tokenizer,
                max_tokens=tb_max_tokens,
                num_layers=tb_num_layers,
                threshold=tb_threshold,
                top_k=tb_top_k,
                selection_mode=tb_selection_mode,
                summarization_length=tb_summarization_length,
                summarization_model=tb_summarization_model,
                embedding_models=tb_embedding_models,
                cluster_embedding_model=tb_cluster_embedding_model,
            )

        elif not isinstance(tree_builder_config, tree_builder_config_class):
            raise ValueError(
                f"tree_builder_config must be a direct instance of {tree_builder_config_class} for tree_builder_type '{tree_builder_type}'"
            )

        # Set TreeRetrieverConfig
        if tree_retriever_config is None:
            tree_retriever_config = TreeRetrieverConfig(
                tokenizer=tr_tokenizer,
                threshold=tr_threshold,
                top_k=tr_top_k,
                selection_mode=tr_selection_mode,
                context_embedding_model=tr_context_embedding_model,
                embedding_model=tr_embedding_model,
                num_layers=tr_num_layers,
                start_layer=tr_start_layer,
            )
        elif not isinstance(tree_retriever_config, TreeRetrieverConfig):
            raise ValueError(
                "tree_retriever_config must be an instance of TreeRetrieverConfig"
            )

        # Assign the created configurations to the instance
        self.tree_builder_config = tree_builder_config
        self.tree_retriever_config = tree_retriever_config
        self.qa_model = qa_model or GPT3TurboQAModel()
        self.tree_builder_type = tree_builder_type

    def log_config(self):
        config_summary = """
        RetrievalAugmentationConfig:
            {tree_builder_config}
            
            {tree_retriever_config}
            
            QA Model: {qa_model}
            Tree Builder Type: {tree_builder_type}
        """.format(
            tree_builder_config=self.tree_builder_config.log_config(),
            tree_retriever_config=self.tree_retriever_config.log_config(),
            qa_model=self.qa_model,
            tree_builder_type=self.tree_builder_type,
        )
        return config_summary


class RetrievalAugmentation:
    """
    Main RAPTOR class that manages the entire process.
    
    Process Flow:
    1. Initialization: Set up configurations
    2. Document Addition: Process text and build tree
    3. Retrieval: Search tree and answer questions
    """

    def __init__(self, config=None, tree=None):
        """
        Step 1: Initialize RAPTOR
        - Set up configurations
        - Initialize tree builder
        - Prepare retriever and QA model
        """
        if config is None:
            config = RetrievalAugmentationConfig()
        if not isinstance(config, RetrievalAugmentationConfig):
            raise ValueError(
                "config must be an instance of RetrievalAugmentationConfig"
            )

        # Check if tree is a string (indicating a path to a pickled tree)
        if isinstance(tree, str):
            try:
                with open(tree, "rb") as file:
                    self.tree = pickle.load(file)
                if not isinstance(self.tree, Tree):
                    raise ValueError("The loaded object is not an instance of Tree")
            except Exception as e:
                raise ValueError(f"Failed to load tree from {tree}: {e}")
        elif isinstance(tree, Tree) or tree is None:
            self.tree = tree
        else:
            raise ValueError(
                "tree must be an instance of Tree, a path to a pickled Tree, or None"
            )

        tree_builder_class = supported_tree_builders[config.tree_builder_type][0]
        self.tree_builder = tree_builder_class(config.tree_builder_config)

        self.tree_retriever_config = config.tree_retriever_config
        self.qa_model = config.qa_model

        if self.tree is not None:
            self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)
        else:
            self.retriever = None

        logging.info(
            f"Successfully initialized RetrievalAugmentation with Config {config.log_config()}"
        )

    def add_documents(self, docs):
        """
        Step 2: Document Processing and Tree Construction
        
        Process:
        1. Check if tree exists (warn about overwriting)
        2. Build new tree from text:
           - Split text into chunks
           - Create leaf nodes
           - Build tree structure
        3. Set up retriever for the new tree
        """
        if self.tree is not None:
            user_input = input(
                "Warning: Overwriting existing tree. Did you mean to call 'add_to_existing' instead? (y/n): "
            )
            if user_input.lower() == "y":
                # self.add_to_existing(docs)
                return

        self.tree = self.tree_builder.build_from_text(text=docs)
        self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)

    def retrieve(
        self,
        question,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = True,
    ):
        """
        Step 3: Information Retrieval
        
        Process:
        1. Take question input
        2. Navigate tree structure:
           - Start from specified layer
           - Find similar nodes
           - Traverse down through relevant branches
        3. Collect relevant context
        4. Return information for QA
        """
        if self.retriever is None:
            raise ValueError(
                "The TreeRetriever instance has not been initialized. Call 'add_documents' first."
            )

        return self.retriever.retrieve(
            question,
            start_layer,
            num_layers,
            top_k,
            max_tokens,
            collapse_tree,
            return_layer_information,
        )

    def answer_question(
        self,
        question,
        top_k: int = 10,
        start_layer: int = None,
        num_layers: int = None,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
    ):
        """
        Step 4: Question Answering
        
        Process:
        1. Retrieve relevant context using tree
        2. Format question and context
        3. Send to QA model
        4. Return generated answer
        """
        # if return_layer_information:
        context, layer_information = self.retrieve(
            question, start_layer, num_layers, top_k, max_tokens, collapse_tree, True
        )

        answer = self.qa_model.answer_question(context, question)

        if return_layer_information:
            return answer, layer_information

        return answer

    def save(self, path):
        if self.tree is None:
            raise ValueError("There is no tree to save.")
        with open(path, "wb") as file:
            pickle.dump(self.tree, file)
        logging.info(f"Tree successfully saved to {path}")
