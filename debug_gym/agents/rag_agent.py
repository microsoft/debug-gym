import json

import numpy as np

from debug_gym.agents.base_agent import BaseAgent, register_agent
from debug_gym.agents.experience_loader import (
    ExperienceDataset,
    load_experience_from_file,
)
from debug_gym.agents.utils import FaissRetriever, SentenceEncoder
from debug_gym.gym.utils import filter_non_utf8


@register_agent
class RAGAgent(BaseAgent):
    name = "rag_agent"
    system_prompt = "You are a debugging agent specialized in fixing Python programs. Your goal is to debug a Python program to make sure it can pass a set of test functions. You have access to a set of tools including the pdb debugger to help you investigate the code before proposing a patch. While the code may seem familiar to you from your training, you should not assume you know the code. Instead, you must use the pdb debugger to investigate the code and understand the potential bugs. A common debugging workflow is to 1) find suspicious files and lines (from error messages or test failures); 2) set breakpoints at suspicious places; 3) continue execution so the frame is at the breakpoint you set; 4) then print necessary values to identify the bugs. Once you have gained enough information, propose a rewriting patch to fix the bugs. Avoid rewriting the entire code, focus on the bugs only. You can only call one tool at a time. Do not repeat your previous action, especially if it returned tool calling errors or it resulted in information that you already know. You can think step by step to help you make the decision at every step, but you must be concise and avoid overthinking. If you are confident that you have enough information, propose a patch to fix the bugs by calling the rewrite tool. If you are not sure, continue using the pdb tool to gather more information before proposing a patch. After every rewrite, it's always a good idea to call the eval tool to execute the new code and check if it passes the tests; if it does not, the tool will return the error messages, which you can use to continue debugging. Output both your thinking process (if any) and the tool call in the response. "

    def __init__(
        self,
        config: dict,
        env,
        llm=None,
        logger=None,
    ):
        super().__init__(config, env, llm, logger)

        # Initialize configuration parameters
        self.rag_num_retrievals = self.config.get(
            "rag_num_retrievals", 1
        )  # how many examples to retrieve
        self.rag_indexing_method = self.parse_indexing_method(
            self.config.get("rag_indexing_method", None)
        )  # how to index the conversation history
        self.sentence_encoder_model = self.config.get(
            "sentence_encoder_model", "Qwen/Qwen3-Embedding-0.6B"
        )

        # Initialize RAG components if dataset is provided
        experience_trajectory_path = self.config.get("experience_trajectory_path", None)
        assert (
            experience_trajectory_path is not None
        ), "Experience path must be provided in the config"
        self.load_experience_trajectory_from_file(experience_trajectory_path)

        self.encoder = None
        self.retriever = None
        self.data_sentence = None
        self.data_label = None

        if self.dataset is not None:
            self._initialize_rag()

    def parse_indexing_method(self, method: str):
        """Parse the indexing method from the configuration.
        The input string should be in the format of "method-step".
        Step indicates how many assistant-user pairs to use for indexing.
        If step is not provided, it defaults to 1.
        supported methods:
        - observation: use the observation as the query
        - tool_name: use the tool name as the query
        - tool_call: use the entire tool call (including arguments) as the query
        For example, "tool_name-5" means to use the concatenation of the last 5 tool names as the query.
        """
        assert method is not None, "rag_indexing_method must be provided in the config"

        method, step = method.rsplit("-", 1) if "-" in method else (method, 1)
        assert method in [
            "observation",
            "tool_name",
            "tool_call",
        ], f"Invalid rag_indexing_method: {method}. Supported methods: observation, tool_name, tool_call"
        assert (
            step.isdigit()
        ), f"Invalid step value: {step}. It should be a positive integer."
        step = int(step)
        assert step > 0, "Step must be a positive integer."
        return [method, step]

    def load_experience_trajectory_from_file(
        self, file_path: str, max_examples: int = None
    ):
        """Load experience trajectories from a JSONL file."""
        self.experience_trajectories = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if max_examples and line_num > max_examples:
                        break
                    try:
                        experience_json = json.loads(line.strip())
                        # filter out trajectories that failed to meet criteria
                        satisfied_criteria = experience_json.get(
                            "satisfied_criteria", []
                        )
                        if (
                            "follows_proper_debugging_workflow"
                            not in satisfied_criteria
                            or "has_successful_outcome" not in satisfied_criteria
                        ):
                            continue
                        self.experience_trajectories.append(experience_json["messages"])
                    except json.JSONDecodeError:
                        self.logger.warning(f"Skipping invalid JSON on line {line_num}")
        except Exception as e:
            self.logger.error(f"Error loading experience trajectories from file: {e}")

    def _initialize_rag(self):
        """Initialize the RAG components: encoder and retriever."""
        self.logger.info("Initializing RAG components...")

        # Get data from dataset
        self.data_sentence, self.data_label = self.dataset.get_data("train")
        self.logger.info(f"Loaded {len(self.data_sentence)} training examples")

        # Initialize encoder
        self.encoder = SentenceEncoder(model_name=self.sentence_encoder_model)

        # Build index
        self._build_index()

    def _build_index(self):
        """Build the vector index for retrieval."""
        self.logger.info("Building vector index...")

        # Encode all training sentences
        train_sentence_representations = self.encoder.encode_sentence(
            self.data_sentence, batch_size=32
        )

        # Initialize retriever
        encoding_dim = train_sentence_representations.shape[1]
        self.retriever = FaissRetriever(encoding_dim)

        # Add representations to index
        self.retriever.add(train_sentence_representations)
        self.logger.info(
            f"Built index with {len(self.data_sentence)} examples, embedding dim: {encoding_dim}"
        )

    def _retrieve_relevant_examples(self, query_text: str):
        """Retrieve relevant examples based on query text."""
        if self.retriever is None or self.rag_num_retrievals <= 0:
            return [], []

        # Encode the query
        query_representation = self.encoder.encode_sentence([query_text], batch_size=1)[
            0
        ]

        # Retrieve similar examples
        distances, indices = self.retriever.retrieve(
            np.array([query_representation]), topk=self.rag_num_retrievals
        )

        # Extract the examples
        relevant_sentences = []
        relevant_labels = []

        for i, idx in enumerate(indices[0]):
            if idx < len(self.data_sentence):  # Safety check
                relevant_sentences.append(self.data_sentence[idx])
                relevant_labels.append(self.data_label[idx])

        return relevant_sentences, relevant_labels

    def _format_retrieved_examples(self, sentences, labels):
        """Format retrieved examples for inclusion in prompt."""
        if not sentences:
            return ""

        examples_text = "\n\n--- Retrieved Similar Examples ---\n"
        for i, (sentence, label) in enumerate(zip(sentences, labels), 1):
            examples_text += f"\nExample {i}:\n"
            examples_text += f"Context: {sentence}\n"
            examples_text += f"Solution: {label}\n"
        examples_text += "\n--- End of Retrieved Examples ---\n"

        return examples_text

    def build_system_prompt(self, info):
        """Override to include RAG retrieved examples in system prompt."""
        # Get the base system prompt
        base_messages = super().build_system_prompt(info)

        # If RAG is not initialized, return base prompt
        if self.retriever is None:
            return base_messages

        # Create query text from current context
        query_parts = []
        if hasattr(info, "instructions") and info.instructions:
            query_parts.append(info.instructions)
        if hasattr(info, "observation") and info.observation:
            query_parts.append(str(info.observation))

        query_text = " ".join(query_parts)

        # Retrieve relevant examples
        if query_text.strip():
            relevant_sentences, relevant_labels = self._retrieve_relevant_examples(
                query_text
            )
            examples_text = self._format_retrieved_examples(
                relevant_sentences, relevant_labels
            )

            # Add examples to system prompt
            if examples_text and base_messages:
                original_content = base_messages[0]["content"]
                enhanced_content = original_content + "\n" + examples_text
                # Trim if necessary
                enhanced_content = self.trim_message(
                    enhanced_content, max_length_percentage=0.9
                )
                base_messages[0]["content"] = filter_non_utf8(enhanced_content)

        return base_messages

    def set_dataset(self, dataset):
        """Set dataset and reinitialize RAG components."""
        self.dataset = dataset
        if dataset is not None:
            self._initialize_rag()
        else:
            self.encoder = None
            self.retriever = None
            self.data_sentence = None
            self.data_label = None

    @classmethod
    def from_experience_file(
        cls,
        experience_file_path: str,
        config: dict,
        env,
        llm=None,
        logger=None,
        max_examples: int = None,
    ):
        """
        Create a RAG agent from an experience file.

        Args:
            experience_file_path: Path to the JSONL file containing debugging experiences
            config: Agent configuration
            env: Environment instance
            llm: Language model instance
            logger: Logger instance
            max_examples: Maximum number of examples to load from the file

        Returns:
            RAGAgent instance with loaded experiences
        """
        # Create dataset from experience file
        dataset = ExperienceDataset(experience_file_path, max_examples=max_examples)

        # Create and return RAG agent
        return cls(config=config, env=env, llm=llm, logger=logger, dataset=dataset)

    def load_experiences_from_file(self, file_path: str, max_examples: int = None):
        """
        Load experiences from a file and reinitialize RAG components.

        Args:
            file_path: Path to the JSONL file containing debugging experiences
            max_examples: Maximum number of examples to load
        """
        dataset = ExperienceDataset(file_path, max_examples=max_examples)
        self.set_dataset(dataset)
