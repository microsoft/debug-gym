import json
import os
import pickle
import re

import numpy as np

from debug_gym.agents.base_agent import register_agent
from debug_gym.agents.debug_agent import DebugAgent
from debug_gym.agents.encoding_service import EncodingServiceClient
from debug_gym.agents.shared_cache import get_shared_cache_manager
from debug_gym.agents.utils import FaissRetriever, SentenceEncoder
from debug_gym.gym.utils import filter_non_utf8


@register_agent
class RAGAgent(DebugAgent):
    """
    RAG (Retrieval-Augmented Generation) Agent that uses cached embeddings for efficiency.

    Cache configuration options:
    - rag_cache_dir: Directory to store cached embeddings (default: ".rag_cache")
    - rag_use_cache: Whether to use caching (default: True)
    - rag_use_encoding_service: Whether to use the encoding service (default: True)
    - rag_encoding_service_host: Host for encoding service (default: "localhost")
    - rag_encoding_service_port: Port for encoding service (default: 8765)

    The agent will automatically cache computed embeddings based on:
    - Experience trajectory file path and modification time
    - RAG indexing method
    - Sentence encoder model

    For parallel execution efficiency:
    - Uses shared cache manager to avoid loading multiple copies of embeddings
    - Can use encoding service to avoid loading multiple copies of the model
    """

    name = "rag_agent"
    delimiter = " <STEP_DELIMITER> "

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
        # Cache directory for storing computed representations
        self.cache_dir = self.config.get("rag_cache_dir", ".rag_cache")
        self.use_cache = self.config.get("rag_use_cache", True)

        # Encoding service configuration
        self.use_encoding_service = self.config.get("rag_use_encoding_service", True)
        self.encoding_service_host = self.config.get(
            "rag_encoding_service_host", "localhost"
        )
        self.encoding_service_port = self.config.get("rag_encoding_service_port", 8765)
        self.encoding_service_timeout = self.config.get(
            "rag_encoding_service_timeout", 120
        )

        # Initialize shared cache manager
        if self.use_cache:
            self.cache_manager = get_shared_cache_manager(self.cache_dir)
        else:
            self.cache_manager = None

        self.experience_trajectory_path = self.config.get(
            "experience_trajectory_path", None
        )
        assert (
            self.experience_trajectory_path is not None
        ), "Experience path must be provided in the config"
        # Load experience trajectories from file
        self.load_experience_trajectory_from_file(self.experience_trajectory_path)
        # Build retrieval dataset
        self.build_retrieval_dataset()
        # Initialize encoder (either service client or local)
        self._initialize_encoder()
        # Build index
        self._build_index()

    def parse_indexing_method(self, method: str):
        """Parse the indexing method from the configuration.
        The input string should be in the format of "method-step".
        Step indicates how many assistant-user pairs to use for indexing.
        If step is not provided, it defaults to 1.
        supported methods:
        - observation: use the observation (user or tool response) as the query
        - tool_name: use the tool name as the query
        - tool_call: use the entire tool call (including arguments) as the query
        - tool_call_with_reasoning: use the tool call with reasoning as the query
        For example, "tool_name-5" means to use the concatenation of the last 5 tool names as the query.
        """
        assert method is not None, "rag_indexing_method must be provided in the config"

        method, step = method.rsplit("-", 1) if "-" in method else (method, "1")
        assert method in [
            "observation",
            "tool_name",
            "tool_call",
            "tool_call_with_reasoning",
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

    def build_retrieval_dataset(self):
        """Build a dataset for retrieval based on the loaded experience trajectories and the indexing method.
        For example, given a trajectory of messages:
        [sys, user, assistant1, tool1, assistant2, tool2, user, assistant3],
        if method=tool_call, and step=2, the dataset will contain:
        input: assistant1; label: assistant2, (when there are less than 2 step, we use all the available steps)
        input: assistant2; label: assistant3,
        input: assistant1, assistant2; label: assistant3,
        """

        def find_last_k_messages_with_role(trajectory, role, k):
            """Find the last k messages with the specified role in the trajectory."""
            if isinstance(role, str):
                role = [role]
            messages = [msg for msg in trajectory if msg["role"] in role]
            return messages[-k:] if len(messages) >= k else messages

        method, step = self.rag_indexing_method
        self.data_input, self.data_label = [], []

        for trajectory in self.experience_trajectories:
            for i in range(len(trajectory)):
                # skip non-assistant messages because assistant messages are the labels
                if trajectory[i]["role"] != "assistant":
                    continue
                # skip the assistant message if it does not have a tool call
                if "tool_calls" not in trajectory[i] or not trajectory[i]["tool_calls"]:
                    continue
                if (
                    "function" not in trajectory[i]["tool_calls"][0]
                    or not trajectory[i]["tool_calls"][0]["function"]
                ):
                    continue
                _label = {"tool_calls": trajectory[i]["tool_calls"][0]["function"]}
                if "content" in trajectory[i]:
                    _label["content"] = trajectory[i]["content"]
                label = json.dumps(_label)
                for __step in range(1, step + 1):
                    match method:
                        case "observation":
                            input_list = find_last_k_messages_with_role(
                                trajectory[:i], ["user", "tool"], __step
                            )
                            if not input_list:
                                continue
                            input_list = [msg["content"] for msg in input_list]
                            input = self.delimiter.join(input_list)
                        case "tool_name":
                            input_list = find_last_k_messages_with_role(
                                trajectory[:i], "assistant", __step
                            )
                            if not input_list:
                                continue
                            tool_name_list = []
                            for msg in input_list:
                                if "tool_calls" in msg and msg["tool_calls"]:
                                    if (
                                        "function" in msg["tool_calls"][0]
                                        and msg["tool_calls"][0]["function"]
                                    ):
                                        tool_name = msg["tool_calls"][0].get("name", "")
                                        if tool_name:
                                            tool_name_list.append(tool_name)
                            if not tool_name_list:
                                continue
                            input = self.delimiter.join(tool_name_list)
                        case "tool_call":
                            input_list = find_last_k_messages_with_role(
                                trajectory[:i], "assistant", __step
                            )
                            if not input_list:
                                continue
                            tool_call_list = []
                            for msg in input_list:
                                if "tool_calls" in msg and msg["tool_calls"]:
                                    if (
                                        "function" in msg["tool_calls"][0]
                                        and msg["tool_calls"][0]["function"]
                                    ):
                                        tool_call = json.dumps(
                                            msg["tool_calls"][0]["function"]
                                        )
                                        tool_call_list.append(tool_call)
                            if not tool_call_list:
                                continue
                            input = self.delimiter.join(tool_call_list)
                        case "tool_call_with_reasoning":
                            input_list = find_last_k_messages_with_role(
                                trajectory[:i], "assistant", __step
                            )
                            if not input_list:
                                continue
                            tool_call_with_reasoning_list = []
                            for msg in input_list:
                                tmp = {}
                                if "tool_calls" in msg and msg["tool_calls"]:
                                    if (
                                        "function" in msg["tool_calls"][0]
                                        and msg["tool_calls"][0]["function"]
                                    ):
                                        tmp["tool_calls"] = msg["tool_calls"][0][
                                            "function"
                                        ]
                                if "content" in msg:
                                    tmp["content"] = msg["content"]
                                if tmp:
                                    tool_call_with_reasoning_list.append(
                                        json.dumps(tmp)
                                    )
                            if not tool_call_with_reasoning_list:
                                continue
                            input = self.delimiter.join(tool_call_with_reasoning_list)
                        case _:
                            raise ValueError(
                                f"Invalid rag_indexing_method: {method}. Supported methods: observation, tool_name, tool_call, tool_call_with_reasoning"
                            )
                    self.data_input.append(filter_non_utf8(input))
                    self.data_label.append(filter_non_utf8(label))
        self.logger.info(
            f"Built retrieval dataset with {len(self.data_input)} examples using method: {method}, max step: {step}"
        )

    def _initialize_encoder(self):
        """Initialize encoder (either service client or local instance)."""
        if self.use_encoding_service:
            self.encoder_client = EncodingServiceClient(
                host=self.encoding_service_host,
                port=self.encoding_service_port,
                timeout=self.encoding_service_timeout,
            )

            # Check if service is available
            if self.encoder_client.is_service_available():
                self.logger.info(
                    f"Using encoding service at {self.encoding_service_host}:{self.encoding_service_port}"
                )
                self.encoder = self.encoder_client
            else:
                self.logger.warning(
                    f"Encoding service not available at {self.encoding_service_host}:{self.encoding_service_port}, "
                    "falling back to local encoder"
                )
                self.use_encoding_service = False
                self.encoder = SentenceEncoder(model_name=self.sentence_encoder_model)
        else:
            self.logger.info("Using local sentence encoder")
            self.encoder = SentenceEncoder(model_name=self.sentence_encoder_model)

    def _generate_cache_key(self):
        """Generate a human-readable cache key based on trajectory path, indexing method, and encoder model."""
        # Extract filename from trajectory path
        trajectory_filename = os.path.basename(self.experience_trajectory_path)
        if trajectory_filename.endswith(".jsonl"):
            trajectory_filename = trajectory_filename[:-6]  # Remove .jsonl extension

        # Create indexing method string
        method, step = self.rag_indexing_method
        indexing_str = f"{method}-{step}"

        # Extract model name (last part after /)
        model_name = (
            self.sentence_encoder_model.split("/")[-1]
            if "/" in self.sentence_encoder_model
            else self.sentence_encoder_model
        )

        # Sanitize strings for filename safety
        def sanitize_for_filename(s):
            # Replace problematic characters with underscores
            return re.sub(r"[^\w\-.]", "_", s)

        trajectory_clean = sanitize_for_filename(trajectory_filename)
        indexing_clean = sanitize_for_filename(indexing_str)
        model_clean = sanitize_for_filename(model_name)

        # Create interpretable cache key
        cache_key = f"{trajectory_clean}_{indexing_clean}_{model_clean}"
        return cache_key

    def _build_index(self):
        """Build the vector index for retrieval with shared caching support."""
        self.logger.info("Building vector index...")

        input_representations = None

        # Use shared cache manager if caching is enabled
        if self.use_cache and self.cache_manager:
            cache_key = self._generate_cache_key()

            def compute_embeddings(data_input):
                """Callback function to compute embeddings."""
                return self.encoder.encode_sentence(data_input, batch_size=16)

            # Use shared cache manager
            self.data_input, input_representations = (
                self.cache_manager.load_or_create_cache(
                    cache_key=cache_key,
                    indexing_method=self.rag_indexing_method,
                    encoder_model=self.sentence_encoder_model,
                    data_input=self.data_input,
                    compute_callback=compute_embeddings,
                )
            )
        else:
            # Compute representations without caching
            self.logger.info(
                "Computing input representations (this may take time with GPU)..."
            )
            input_representations = self.encoder.encode_sentence(
                self.data_input, batch_size=16
            )

        # Initialize retriever
        encoding_dim = input_representations.shape[1]
        self.retriever = FaissRetriever(encoding_dim)

        # Add representations to index
        self.retriever.add(input_representations)
        self.logger.info(
            f"Built index with {len(self.data_input)} examples, embedding dim: {encoding_dim}"
        )

    def _retrieve_relevant_examples(self, query_text: str):
        """Retrieve relevant examples based on query text.
        The query text is converted from the the agent's history based on the indexing method.
        """
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
        relevant_inputs, relevant_labels = [], []

        for i, idx in enumerate(indices[0]):
            if idx < len(self.data_input):  # Safety check
                relevant_inputs.append(self.data_input[idx])
                relevant_labels.append(self.data_label[idx])

        return relevant_inputs, relevant_labels

    def extract_query_text_from_history(self):
        """Extract the query text from the agent's history based on the indexing method."""
        method, step = self.rag_indexing_method
        history, _ = self.history.get()  # list[EnvInfo]
        history = history[-step:]
        if len(history) == 0:
            return None
        match method:
            case "observation":
                observation_list = [
                    item.step_observation.observation for item in history
                ]
                if not observation_list:
                    return None
                query_text = self.delimiter.join(observation_list)
            case "tool_name":
                tool_name_list = [item.action.name for item in history if item.action]
                if not tool_name_list:
                    return None
                query_text = self.delimiter.join(tool_name_list)
            case "tool_call":
                tool_call_list = [
                    json.dumps(
                        {"name": item.action.name, "arguments": item.action.arguments}
                    )
                    for item in history
                    if item.action
                ]
                if not tool_call_list:
                    return None
                query_text = self.delimiter.join(tool_call_list)
            case "tool_call_with_reasoning":
                tool_call_with_reasoning_list = []
                for item in history:
                    _tmp = {}
                    if item.action:
                        _tmp["tool_calls"] = {
                            "name": item.action.name,
                            "arguments": item.action.arguments,
                        }
                    if item.action_reasoning:
                        _tmp["content"] = item.action_reasoning
                    if not _tmp:
                        continue
                    tool_call_with_reasoning_list.append(json.dumps(_tmp))
                if not tool_call_with_reasoning_list:
                    return None
                query_text = self.delimiter.join(tool_call_with_reasoning_list)
            case _:
                raise ValueError(
                    f"Invalid rag_indexing_method: {method}. Supported methods: observation, tool_name, tool_call, tool_call_with_reasoning"
                )
        return filter_non_utf8(query_text)

    def build_question_prompt(self):
        # Extract the query text from the history
        query_text = self.extract_query_text_from_history()
        if query_text is None:
            return []
        # Retrieve relevant examples
        _, relevant_examples = self._retrieve_relevant_examples(query_text)
        if not relevant_examples:
            self.logger.warning(
                "No relevant examples found for the current query. Proceeding without RAG."
            )
            return []

        # Build the question prompt with retrieved examples
        content = "I have retrieved some relevant examples to help you make a decision. Note that these examples are not guaranteed to be correct or applicable to the current situation, but you can use them as references if you are unsure about the next step. "
        content += "You can ignore the examples that are not relevant to the current situation. Here are the examples:\n"
        deduplicate = set()
        for example in relevant_examples:
            _ex = json.dumps(example, indent=2)
            if _ex in deduplicate:
                continue
            content += f"\nExample {len(deduplicate) + 1}:\n{_ex}\n"
            deduplicate.add(_ex)

        # debug_gym_ignore is used to prevent the history tracker from saving this message
        # so that we don't have to record the retrieved examples after every step in the history
        messages = [{"role": "user", "content": content, "debug_gym_ignore": True}]
        return messages
