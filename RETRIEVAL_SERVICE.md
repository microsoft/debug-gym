# Retrieval as a Service

This document describes how to use the new retrieval service functionality that enables sharing retrieval indexes across multiple RAG agents.

## Overview

The retrieval service allows multiple RAG agents to share the same vector index and retrieval logic, avoiding the need to load multiple copies of large indexes in memory. This is particularly useful for parallel execution scenarios.

## Architecture

```
┌─────────────┐    ┌─────────────────────┐
│   RAG Agent │───▶│  Retrieval Service  │
│             │    │                     │
│ - Extracts  │    │ - Manages indexes   │
│   queries   │    │ - Handles retrieval │
│ - Builds    │    │ - Sentence encoding │
│   prompts   │    │ - Caching           │
└─────────────┘    └─────────────────────┘
```

## Services

### Retrieval Service
Manages vector indexes, handles retrieval requests, and performs sentence encoding internally.

**Default port:** 8766

**Start command:**
```bash
python scripts/start_retrieval_service.py --port 8766 --config scripts/config_retrieval_service.yaml
```

## Configuration

### RAG Agent Configuration

Update your agent configuration to use the retrieval service:

```yaml
rag_agent:
    # Basic RAG settings
    rag_num_retrievals: 3
    rag_indexing_method: "tool_call_with_reasoning-3"
    rag_indexing_batch_size: 16
    sentence_encoder_model: "Qwen/Qwen3-Embedding-0.6B"
    experience_trajectory_path: "path/to/your/experience.jsonl"
    
    # Retrieval service configuration
    rag_use_retrieval_service: true
    rag_retrieval_service_host: "localhost"
    rag_retrieval_service_port: 8766
    rag_retrieval_service_timeout: 300
    
    # Cache settings
    rag_cache_dir: ".rag_cache"
    rag_use_cache: true
```

### Retrieval Service Configuration

Create a configuration file for the retrieval service:

```yaml
# config_retrieval_service.yaml
rag_cache_dir: ".rag_cache"
rag_use_cache: true
sentence_encoder_model: "Qwen/Qwen3-Embedding-0.6B"
```

## Usage Workflow

### 1. Start the Retrieval Service

```bash
python scripts/start_retrieval_service.py --config scripts/config_retrieval_service.yaml
```

### 2. Run RAG Agents

The RAG agents will automatically:
1. Connect to the retrieval service
2. Build indexes (if not already built)
3. Retrieve relevant examples during execution

```bash
python scripts/run.py --config scripts/config_swesmith.yaml --agent rag_agent
```

## API Endpoints

### Retrieval Service

- `GET /health` - Health check
- `GET /indexes` - List available indexes
- `POST /build_index` - Build a new index
- `POST /retrieve` - Retrieve relevant examples

### Build Index Request

```json
{
    "index_key": "unique_index_identifier",
    "experience_trajectory_path": "path/to/experience.jsonl",
    "rag_indexing_method": "tool_call_with_reasoning-3",
    "sentence_encoder_model": "Qwen/Qwen3-Embedding-0.6B",
    "rag_indexing_batch_size": 16,
    "use_cache": true
}
```

### Retrieve Request

```json
{
    "index_key": "unique_index_identifier",
    "query_text": "text to find similar examples for",
    "num_retrievals": 3
}
```

## Benefits

1. **Memory Efficiency**: Only one copy of the index is loaded in memory
2. **Faster Startup**: Agents don't need to rebuild indexes individually
3. **Scalability**: Multiple agents can share the same retrieval infrastructure
4. **Caching**: Shared cache across all agents using the same index
5. **Service Isolation**: Retrieval logic is separated from agent logic

## Migration from Local Retrieval

The new retrieval service is designed to be a drop-in replacement for the local retrieval logic. Simply:

1. Start the retrieval service
2. Update your configuration to set `rag_use_retrieval_service: true`
3. Run your RAG agents as usual

The agents will automatically connect to the service and behave identically to the local retrieval implementation.

## Troubleshooting

### Service Connection Issues

- Ensure the retrieval service is running and accessible
- Check that the host and port configuration matches
- Verify firewall settings if running across different machines

### Index Building Failures

- Check that the experience trajectory file exists and is readable
- Verify that the encoding service is available (if using encoding as a service)
- Check the service logs for detailed error messages

### Performance Issues

- Consider adjusting batch sizes for encoding
- Monitor memory usage of the retrieval service
- Use caching to avoid recomputing embeddings
