import json
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from debug_gym.agents.utils import FaissRetriever, SentenceEncoder


class TestSentenceEncoder:
    """Test cases for the SentenceEncoder class."""

    @patch("debug_gym.agents.utils.SentenceTransformer")
    def test_init_default_model(self, mock_sentence_transformer):
        """Test SentenceEncoder initialization with default model."""
        encoder = SentenceEncoder()
        mock_sentence_transformer.assert_called_once_with("Qwen/Qwen3-Embedding-0.6B")

    @patch("debug_gym.agents.utils.SentenceTransformer")
    def test_init_custom_model(self, mock_sentence_transformer):
        """Test SentenceEncoder initialization with custom model."""
        custom_model = "custom/model-name"
        encoder = SentenceEncoder(model_name=custom_model)
        mock_sentence_transformer.assert_called_once_with(custom_model)

    @patch("debug_gym.agents.utils.SentenceTransformer")
    def test_encode_sentence_default_batch_size(self, mock_sentence_transformer):
        """Test encoding sentences with default batch size."""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        # Mock the encode method to return dummy embeddings
        expected_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.encode.return_value = expected_embeddings

        encoder = SentenceEncoder()
        sentences = ["Hello world", "Test sentence"]

        result = encoder.encode_sentence(sentences)

        mock_model.encode.assert_called_once_with(
            sentences, batch_size=32, convert_to_numpy=True
        )
        np.testing.assert_array_equal(result, expected_embeddings)

    @patch("debug_gym.agents.utils.SentenceTransformer")
    def test_encode_sentence_custom_batch_size(self, mock_sentence_transformer):
        """Test encoding sentences with custom batch size."""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        expected_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_model.encode.return_value = expected_embeddings

        encoder = SentenceEncoder()
        sentences = ["Sentence 1", "Sentence 2"]
        batch_size = 16

        result = encoder.encode_sentence(sentences, batch_size=batch_size)

        mock_model.encode.assert_called_once_with(
            sentences, batch_size=batch_size, convert_to_numpy=True
        )
        np.testing.assert_array_equal(result, expected_embeddings)

    @patch("debug_gym.agents.utils.SentenceTransformer")
    def test_encode_sentence_empty_list(self, mock_sentence_transformer):
        """Test encoding empty sentence list."""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        expected_embeddings = np.array([])
        mock_model.encode.return_value = expected_embeddings

        encoder = SentenceEncoder()

        result = encoder.encode_sentence([])

        mock_model.encode.assert_called_once_with(
            [], batch_size=32, convert_to_numpy=True
        )
        np.testing.assert_array_equal(result, expected_embeddings)


class TestFaissRetriever:
    """Test cases for the FaissRetriever class."""

    @patch("debug_gym.agents.utils.faiss")
    def test_init(self, mock_faiss):
        """Test FaissRetriever initialization."""
        mock_index = MagicMock()
        mock_faiss.IndexFlatL2.return_value = mock_index

        encoding_dim = 128
        retriever = FaissRetriever(encoding_dim)

        mock_faiss.IndexFlatL2.assert_called_once_with(encoding_dim)
        assert retriever.index == mock_index

    @patch("debug_gym.agents.utils.faiss")
    def test_add_representations(self, mock_faiss):
        """Test adding sentence representations to the index."""
        mock_index = MagicMock()
        mock_faiss.IndexFlatL2.return_value = mock_index

        retriever = FaissRetriever(encoding_dim=3)
        representations = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        retriever.add(representations)

        mock_index.add.assert_called_once_with(representations)

    @patch("debug_gym.agents.utils.faiss")
    def test_retrieve(self, mock_faiss):
        """Test retrieving similar representations."""
        mock_index = MagicMock()
        mock_faiss.IndexFlatL2.return_value = mock_index

        # Mock search results
        expected_distances = np.array([[0.1, 0.3]])
        expected_indices = np.array([[0, 2]])
        mock_index.search.return_value = (expected_distances, expected_indices)

        retriever = FaissRetriever(encoding_dim=3)
        query_representations = np.array([[0.2, 0.3, 0.4]])
        topk = 2

        distances, indices = retriever.retrieve(query_representations, topk)

        mock_index.search.assert_called_once_with(query_representations, topk)
        np.testing.assert_array_equal(distances, expected_distances)
        np.testing.assert_array_equal(indices, expected_indices)

    @patch("debug_gym.agents.utils.faiss")
    def test_retrieve_single_result(self, mock_faiss):
        """Test retrieving single similar representation."""
        mock_index = MagicMock()
        mock_faiss.IndexFlatL2.return_value = mock_index

        # Mock search results for single result
        expected_distances = np.array([[0.05]])
        expected_indices = np.array([[1]])
        mock_index.search.return_value = (expected_distances, expected_indices)

        retriever = FaissRetriever(encoding_dim=2)
        query_representations = np.array([[0.1, 0.2]])
        topk = 1

        distances, indices = retriever.retrieve(query_representations, topk)

        mock_index.search.assert_called_once_with(query_representations, topk)
        np.testing.assert_array_equal(distances, expected_distances)
        np.testing.assert_array_equal(indices, expected_indices)


class TestSentenceEncoderFaissRetrieverIntegration:
    """Integration tests for SentenceEncoder and FaissRetriever."""

    @patch("debug_gym.agents.utils.SentenceTransformer")
    @patch("debug_gym.agents.utils.faiss")
    def test_encode_and_retrieve_workflow(self, mock_faiss, mock_sentence_transformer):
        """Test the complete workflow of encoding and retrieving."""
        # Setup mocks
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        mock_index = MagicMock()
        mock_faiss.IndexFlatL2.return_value = mock_index

        # Mock embeddings for training sentences
        train_embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        mock_model.encode.side_effect = [train_embeddings, np.array([[0.15, 0.25]])]

        # Mock retrieval results
        mock_index.search.return_value = (np.array([[0.05]]), np.array([[0]]))

        # Setup encoder and retriever
        encoder = SentenceEncoder()

        # Encode training sentences
        train_sentences = ["sentence 1", "sentence 2", "sentence 3"]
        encoded_sentences = encoder.encode_sentence(train_sentences)

        # Initialize retriever and add embeddings
        retriever = FaissRetriever(encoding_dim=2)
        retriever.add(encoded_sentences)

        # Encode query and retrieve
        query_sentence = ["similar to sentence 1"]
        query_embedding = encoder.encode_sentence(query_sentence)
        distances, indices = retriever.retrieve(query_embedding, topk=1)

        # Verify calls
        assert mock_model.encode.call_count == 2
        mock_index.add.assert_called_once_with(train_embeddings)
        mock_index.search.assert_called_once()

        np.testing.assert_array_equal(distances, np.array([[0.05]]))
        np.testing.assert_array_equal(indices, np.array([[0]]))
