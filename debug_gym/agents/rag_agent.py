import json
import os
import re

from debug_gym.agents.base_agent import register_agent
from debug_gym.agents.debug_agent import DebugAgent
from debug_gym.gym.utils import filter_non_utf8

# Import from standalone retrieval service
try:
    from retrieval_service.client import RetrievalServiceClient

    RETRIEVAL_SERVICE_AVAILABLE = True
except ImportError:
    RetrievalServiceClient = None
    RETRIEVAL_SERVICE_AVAILABLE = False


@register_agent
class RAGAgent(DebugAgent):
    """
    RAG (Retrieval-Augmented Generation) Agent that uses a retrieval service for efficiency.

    Retrieval service configuration options:

    - rag_retrieval_service_host: Host for retrieval service (default: "localhost")
    - rag_retrieval_service_port: Port for retrieval service (default: 8766)
    - rag_retrieval_service_timeout: Timeout for retrieval service requests (default: 120)

    The agent will communicate with the retrieval service to:
    - Build indexes from experience trajectory files
    - Retrieve relevant examples for the current query

    For parallel execution efficiency:
    - Uses retrieval service to avoid loading multiple copies of indexes
    - Shares retrieval logic across multiple agent instances
    """

    name = "rag_agent"
    delimiter = " <STEP_DELIMITER> "

    def _is_retrieval_service_available(self):
        """Check if retrieval service is available. Can be mocked for testing."""
        return RETRIEVAL_SERVICE_AVAILABLE

    def __init__(
        self,
        config: dict,
        env,
        llm=None,
        logger=None,
    ):
        # Check if retrieval service is available before proceeding
        if not self._is_retrieval_service_available():
            raise ImportError(
                "The standalone retrieval service is required for RAG functionality. "
                "Please install it by running: pip install retrieval-service"
            )

        super().__init__(config, env, llm, logger)

        # Initialize configuration parameters
        self.rag_num_retrievals = self.config.get(
            "rag_num_retrievals", 1
        )  # how many examples to retrieve
        self.rag_indexing_method = self.parse_indexing_method(
            self.config.get("rag_indexing_method", None)
        )  # how to index the conversation history
        self.rag_indexing_batch_size = self.config.get("rag_indexing_batch_size", 16)
        self.sentence_encoder_model = self.config.get(
            "sentence_encoder_model", "Qwen/Qwen3-Embedding-0.6B"
        )

        # Cache directory for storing computed representations
        self.cache_dir = self.config.get("rag_cache_dir", ".rag_cache")
        self.use_cache = self.config.get("rag_use_cache", True)

        # Retrieval service configuration
        self.retrieval_service_host = self.config.get(
            "rag_retrieval_service_host", "localhost"
        )
        self.retrieval_service_port = self.config.get(
            "rag_retrieval_service_port", 8766
        )
        self.retrieval_service_timeout = self.config.get(
            "rag_retrieval_service_timeout", 120
        )

        self.experience_trajectory_path = self.config.get(
            "experience_trajectory_path", None
        )
        assert (
            self.experience_trajectory_path is not None
        ), "Experience path must be provided in the config"

        # Initialize retrieval service client
        self._initialize_retrieval_service()

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

    def _initialize_retrieval_service(self):
        """Initialize retrieval service client."""
        self.retrieval_client = RetrievalServiceClient(
            host=self.retrieval_service_host,
            port=self.retrieval_service_port,
            timeout=self.retrieval_service_timeout,
        )

        # Check if service is available
        if not self.retrieval_client.is_service_available():
            self.logger.error(
                f"Retrieval service not available at {self.retrieval_service_host}:{self.retrieval_service_port}. "
                f"Please start the retrieval service first."
            )
            raise RuntimeError("Retrieval service not available")

        self.logger.info(
            f"Using retrieval service at {self.retrieval_service_host}:{self.retrieval_service_port}"
        )

        # Generate index key based on configuration
        self.index_key = self._generate_index_key()

        # Build index on the service
        self._build_index_on_service()

    def _generate_index_key(self):
        """Generate a unique index key based on trajectory path, indexing method, and encoder model."""
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

        # Sanitize strings for key safety
        def sanitize_for_key(s):
            # Replace problematic characters with underscores
            return re.sub(r"[^\w\-.]", "_", s)

        trajectory_clean = sanitize_for_key(trajectory_filename)
        indexing_clean = sanitize_for_key(indexing_str)
        model_clean = sanitize_for_key(model_name)

        # Create interpretable index key
        index_key = f"{trajectory_clean}_{indexing_clean}_{model_clean}"
        return index_key

    def _build_index_on_service(self):
        """Build the index on the retrieval service."""
        # First check if the index already exists
        if self.retrieval_client.check_index(self.index_key):
            self.logger.info(
                f"Index '{self.index_key}' already exists on retrieval service, skipping build"
            )
            return

        self.logger.info(f"Building index '{self.index_key}' on retrieval service...")

        # Reconstruct indexing method string for the service
        method, step = self.rag_indexing_method
        indexing_method_str = f"{method}-{step}"

        success = self.retrieval_client.build_index(
            index_key=self.index_key,
            experience_trajectory_path=os.path.abspath(self.experience_trajectory_path),
            rag_indexing_method=indexing_method_str,
            sentence_encoder_model=self.sentence_encoder_model,
            rag_indexing_batch_size=self.rag_indexing_batch_size,
            use_cache=self.use_cache,
        )

        if not success:
            raise RuntimeError(
                f"Failed to build index '{self.index_key}' on retrieval service"
            )

        self.logger.info(
            f"Successfully built index '{self.index_key}' on retrieval service"
        )

    def _retrieve_relevant_examples(self, query_text: str):
        """Retrieve relevant examples based on query text using the retrieval service."""
        if self.rag_num_retrievals <= 0:
            return []

        try:
            relevant_examples = self.retrieval_client.retrieve(
                index_key=self.index_key,
                query_text=query_text,
                num_retrievals=self.rag_num_retrievals,
            )
            return relevant_examples
        except Exception as e:
            self.logger.error(f"Error retrieving examples: {str(e)}")
            return []

    def extract_query_text_from_history(self):
        """Extract the query text from the agent's history based on the indexing method."""
        method, step = self.rag_indexing_method
        history, _ = self.history.get()  # list[EnvInfo]
        history = history[-step:]
        if len(history) == 0:
            return None
        if method == "observation":
            observation_list = [item.step_observation.observation for item in history]
            if not observation_list:
                return None
            query_text = self.delimiter.join(observation_list)
        elif method == "tool_name":
            tool_name_list = [item.action.name for item in history if item.action]
            if not tool_name_list:
                return None
            query_text = self.delimiter.join(tool_name_list)
        elif method == "tool_call":
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
        elif method == "tool_call_with_reasoning":
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
        else:
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
        relevant_examples = self._retrieve_relevant_examples(query_text)
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
            # Parse the example if it's a JSON string
            if isinstance(example, str):
                try:
                    example_dict = json.loads(example)
                    _ex = json.dumps(example_dict, indent=2)
                except json.JSONDecodeError:
                    _ex = example
            else:
                _ex = json.dumps(example, indent=2)

            if _ex in deduplicate:
                continue
            content += f"\nExample {len(deduplicate) + 1}:\n{_ex}\n"
            deduplicate.add(_ex)

        # debug_gym_ignore is used to prevent the history tracker from saving this message
        # so that we don't have to record the retrieved examples after every step in the history
        messages = [{"role": "user", "content": content, "debug_gym_ignore": True}]
        return messages
