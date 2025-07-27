import numpy as np

from debug_gym.agents.base_agent import BaseAgent, register_agent
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
        self.num_examples = self.config.get("num_examples", 1)
        self.sentence_encoder_type = self.config.get(
            "sentence_encoder", "sentence-transformer"
        )
        self.sentence_encoder_model = self.config.get(
            "sentence_encoder_model", "Qwen/Qwen3-Embedding-0.6B"
        )

        # Initialize RAG components if dataset is provided
        experience_path = self.config.get("experience_path", None)
        assert (
            experience_path is not None
        ), "Experience path must be provided in the config"
        self.experience = self.load_experience_from_file(experience_path)

        self.encoder = None
        self.retriever = None
        self.data_sentence = None
        self.data_label = None

        if self.dataset is not None:
            self._initialize_rag()

    def load_experience_from_file(self, path):
        pass

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
        if self.retriever is None or self.num_examples <= 0:
            return [], []

        # Encode the query
        query_representation = self.encoder.encode_sentence([query_text], batch_size=1)[
            0
        ]

        # Retrieve similar examples
        distances, indices = self.retriever.retrieve(
            np.array([query_representation]), topk=self.num_examples
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
