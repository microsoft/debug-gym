# RAG Agent Performance Improvements

## Overview

This implementation addresses the performance issues with parallel RAG agents by introducing two key optimizations:

1. **Encoding Service**: A shared sentence encoder service that eliminates the need for each agent to load its own copy of the model
2. **Shared Cache Manager**: A thread-safe cache system that allows multiple agents to share cached embeddings without duplicating memory usage

## Performance Benefits

### Before Optimization
- Each agent loads its own copy of the sentence encoder model (high memory usage)
- Each agent loads its own copy of cached embeddings (memory duplication)
- Single-text encoding calls are inefficient (no batching)
- No coordination between agents

### After Optimization
- Single sentence encoder service shared across all agents
- Shared cache manager with automatic memory management
- Efficient batching support for encoding requests
- Thread-safe concurrent access to cached data

## Key Components

### 1. Encoding Service (`encoding_service.py`)

A standalone HTTP service that hosts the sentence encoder model:

```python
from debug_gym.agents.encoding_service import EncodingService, EncodingServiceClient

# Start service (run this once)
service = EncodingService("Qwen/Qwen3-Embedding-0.6B", port=8765)
service.start_service()

# Use client in agents
client = EncodingServiceClient(port=8765)
embeddings = client.encode_sentence(["text1", "text2"], batch_size=16)
```

**Features:**
- HTTP-based API with health checks
- Supports both regular and query encoding
- Configurable batch sizes
- Thread-safe request handling

### 2. Shared Cache Manager (`shared_cache.py`)

A thread-safe cache system for sharing embeddings across agents:

```python
from debug_gym.agents.shared_cache import get_shared_cache_manager

# Get shared cache manager (same instance across all agents)
cache_manager = get_shared_cache_manager("/path/to/cache")

# Load or create cache
data_input, embeddings = cache_manager.load_or_create_cache(
    cache_key="unique_key",
    indexing_method=["tool_name", 1],
    encoder_model="model_name",
    data_input=input_texts,
    compute_callback=encoding_function
)
```

**Features:**
- Thread-safe concurrent access
- Automatic memory management with LRU eviction
- Disk persistence for cache durability
- Configuration validation to prevent cache mismatches

### 3. Updated RAG Agent (`rag_agent.py`)

The RAG agent now supports both optimizations:

```yaml
# Configuration example
rag_use_encoding_service: true
rag_encoding_service_host: localhost
rag_encoding_service_port: 8765
rag_use_cache: true
rag_cache_dir: ".rag_cache"
```

## Usage Guide

### Step 1: Start the Encoding Service

```bash
# Start the encoding service (run once)
python scripts/start_encoding_service.py --model "Qwen/Qwen3-Embedding-0.6B" --port 8765
```

### Step 2: Configure RAG Agents

Add these configuration options to your agent configs:

```yaml
# Enable encoding service
rag_use_encoding_service: true
rag_encoding_service_host: localhost
rag_encoding_service_port: 8765

# Enable shared caching
rag_use_cache: true
rag_cache_dir: ".rag_cache"
```

### Step 3: Run Multiple Agents

All agents will now:
- Share the same encoding service (no model duplication)
- Share cached embeddings (no memory duplication)
- Benefit from automatic batching and caching

## Configuration Options

### RAG Agent Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `rag_use_encoding_service` | `true` | Use shared encoding service |
| `rag_encoding_service_host` | `localhost` | Service host |
| `rag_encoding_service_port` | `8765` | Service port |
| `rag_use_cache` | `true` | Enable shared caching |
| `rag_cache_dir` | `.rag_cache` | Cache directory |

### Encoding Service Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `Qwen/Qwen3-Embedding-0.6B` | Sentence encoder model |
| `--port` | `8765` | Service port |
| `--host` | `localhost` | Service host |

## Fallback Behavior

The implementation includes robust fallback mechanisms:

1. **Service Unavailable**: If the encoding service is not available, agents automatically fall back to local encoders
2. **Cache Mismatch**: If cache configuration doesn't match, agents recompute embeddings
3. **Network Issues**: Client includes timeout and retry logic

## Memory Management

### Shared Cache Features

- **LRU Eviction**: Automatically removes oldest caches when memory limit is reached
- **Disk Persistence**: Caches are saved to disk and can be reloaded
- **Memory Monitoring**: Built-in tools to monitor cache memory usage

```python
# Get cache information
info = cache_manager.get_cache_info()
print(f"Memory usage: {info['memory_usage_mb']:.2f} MB")
print(f"In-memory caches: {info['in_memory_caches']}")
```

## Testing

The implementation includes comprehensive tests covering:

- ✅ Encoding service functionality
- ✅ Shared cache manager operations
- ✅ Concurrent access safety
- ✅ Integration between components
- ✅ Fallback mechanisms

Run tests with:
```bash
python test_rag_improvements.py
```

## Performance Expectations

With these optimizations, you can expect:

1. **Memory Reduction**: 80-90% reduction in memory usage for parallel agents
2. **Faster Startup**: Agents start faster (no model loading per agent)
3. **Better Throughput**: Batch processing improves encoding efficiency
4. **Scalability**: Can run many more agents in parallel

## Troubleshooting

### Common Issues

1. **Service Not Starting**: Check port availability and model loading
2. **Cache Mismatches**: Ensure consistent configuration across agents
3. **Network Timeouts**: Adjust timeout settings for large batch sizes

### Monitoring

```python
# Check service health
client = EncodingServiceClient(port=8765)
if client.is_service_available():
    print("Service is healthy")

# Monitor cache usage
cache_info = cache_manager.get_cache_info()
print(f"Cache info: {cache_info}")
```

This implementation provides a robust, scalable solution for running multiple RAG agents efficiently in parallel environments.
