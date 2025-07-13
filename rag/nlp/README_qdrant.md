# Qdrant Search Implementation

This document describes the Qdrant-based implementation of the search functionality as a drop-in replacement for the original Elasticsearch/OpenSearch-based `Dealer` class.

## Overview

The `QdrantDealer` class in [`qdrant_search.py`](qdrant_search.py) provides the same interface as the original `Dealer` class but uses Qdrant as the vector database backend instead of Elasticsearch/OpenSearch.

## Features

### ✅ Implemented Features

- **Vector Search**: Full vector similarity search using cosine distance
- **Hybrid Search**: Combines vector and text-based similarity scoring
- **Filtering**: Support for knowledge base, document, and metadata filtering
- **Reranking**: Both built-in and model-based reranking capabilities
- **Pagination**: Proper pagination support for large result sets
- **Citation Insertion**: Automatic citation insertion in generated answers
- **Chunk Management**: List and retrieve document chunks
- **Error Handling**: Graceful error handling and logging
- **Type Safety**: Full type annotations for better code quality

### ⚠️ Limitations

- **Full-text Search**: Limited full-text search capabilities (Qdrant is primarily vector-based)
- **SQL Queries**: SQL retrieval not supported (returns empty results)
- **Tag Operations**: Simplified tag-based content and query operations
- **Aggregations**: Basic aggregation support (document counts)

## Installation

```bash
# Install Qdrant client
pip install qdrant-client

# Optional: Install Qdrant server locally using Docker
docker run -p 6333:6333 qdrant/qdrant
```

## Usage

### Basic Usage

```python
from qdrant_client import QdrantClient
from rag.nlp.qdrant_search import QdrantDealer

# Initialize Qdrant client
client = QdrantClient(url="http://localhost:6333")  # or remote URL

# Create QdrantDealer (same interface as original Dealer)
dealer = QdrantDealer(client)

# Use exactly like the original Dealer
result = dealer.retrieval(
    question="What is machine learning?",
    embd_mdl=your_embedding_model,
    tenant_ids=["tenant_123"],
    kb_ids=["kb_456"],
    page=1,
    page_size=20,
    similarity_threshold=0.2,
    vector_similarity_weight=0.7
)

# Process results same as before
for chunk in result["chunks"]:
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Content: {chunk['content_with_weight']}")
    print(f"Similarity: {chunk['similarity']}")
```

### Drop-in Replacement

```python
# Original code
from rag.nlp.search import Dealer
dealer = Dealer(dataStore)

# Replace with Qdrant version
from rag.nlp.qdrant_search import QdrantDealer
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
dealer = QdrantDealer(client)

# All existing code works the same!
```

## API Reference

### Core Methods

#### `search(req, idx_names, kb_ids, emb_mdl=None, highlight=False, rank_feature=None)`
Main search method that handles both vector and text-based search.

**Parameters:**
- `req`: Search request dictionary
- `idx_names`: Index names (collection names in Qdrant)
- `kb_ids`: Knowledge base IDs for filtering
- `emb_mdl`: Embedding model for vector search
- `highlight`: Enable result highlighting
- `rank_feature`: Ranking features for scoring

**Returns:** `SearchResult` object with results

#### `retrieval(question, embd_mdl, tenant_ids, kb_ids, page, page_size, ...)`
High-level retrieval method with reranking and pagination.

**Parameters:**
- `question`: Search query string
- `embd_mdl`: Embedding model
- `tenant_ids`: Tenant IDs
- `kb_ids`: Knowledge base IDs
- `page`: Page number (1-based)
- `page_size`: Results per page
- `similarity_threshold`: Minimum similarity score
- `vector_similarity_weight`: Weight for vector vs text similarity
- `top`: Maximum results to retrieve before reranking
- `doc_ids`: Optional document ID filter
- `aggs`: Enable aggregations
- `rerank_mdl`: Optional reranking model
- `highlight`: Enable highlighting
- `rank_feature`: Ranking features

**Returns:** Dictionary with `total`, `chunks`, and `doc_aggs`

#### `rerank(sres, query, tkweight=0.3, vtweight=0.7, cfield="content_ltks", rank_feature=None)`
Rerank search results using hybrid similarity.

#### `rerank_by_model(rerank_mdl, sres, query, ...)`
Rerank search results using an external reranking model.

### Utility Methods

#### `chunk_list(doc_id, tenant_id, kb_ids, max_count=1024, offset=0, fields=None)`
List chunks for a specific document.

#### `all_tags(tenant_id, kb_ids, S=1000)`
Get all tags from the collection (simplified implementation).

#### `insert_citations(answer, chunks, chunk_v, embd_mdl, tkweight=0.1, vtweight=0.9)`
Insert citations into generated answers.

## Data Structure

### SearchResult

```python
@dataclass
class SearchResult:
    total: int                              # Total number of results
    ids: List[str]                         # Result IDs
    query_vector: Optional[List[float]]    # Query embedding vector
    field: Optional[Dict[str, Dict]]       # Field data for each result
    highlight: Optional[Dict[str, str]]    # Highlighted content
    aggregation: Optional[Union[List, Dict]] # Aggregation results
    keywords: Optional[List[str]]          # Extracted keywords
    group_docs: Optional[List[List]]       # Grouped documents
```

### Retrieval Result

```python
{
    "total": int,                    # Total matching results
    "chunks": [                      # Result chunks
        {
            "chunk_id": str,
            "content_ltks": str,
            "content_with_weight": str,
            "doc_id": str,
            "docnm_kwd": str,
            "kb_id": str,
            "important_kwd": List[str],
            "image_id": str,
            "similarity": float,
            "vector_similarity": float,
            "term_similarity": float,
            "vector": List[float],
            "positions": List,
            "doc_type_kwd": str,
            "highlight": str            # If highlighting enabled
        }
    ],
    "doc_aggs": [                   # Document aggregations
        {
            "doc_name": str,
            "doc_id": str,
            "count": int
        }
    ]
}
```

## Configuration

### Qdrant Client Configuration

```python
from qdrant_client import QdrantClient

# Local instance
client = QdrantClient(url="http://localhost:6333")

# Remote instance with authentication
client = QdrantClient(
    url="https://your-cluster.qdrant.io",
    api_key="your-api-key"
)

# Custom configuration
client = QdrantClient(
    host="localhost",
    port=6333,
    timeout=60
)
```

### Collection Setup

Before using QdrantDealer, ensure your Qdrant collections are properly configured:

```python
from qdrant_client.models import Distance, VectorParams

# Create collection with appropriate vector size
client.create_collection(
    collection_name="ragflow_tenant_123",
    vectors_config=VectorParams(
        size=384,  # Match your embedding model dimension
        distance=Distance.COSINE
    )
)
```

## Testing

Run the test script to verify the implementation:

```bash
python rag/nlp/test_qdrant_search.py
```

The test script will:
- Compare interfaces between original Dealer and QdrantDealer
- Test basic functionality (with graceful handling if Qdrant server is not available)
- Show usage examples
- Provide implementation summary

## Migration Guide

### From Elasticsearch/OpenSearch to Qdrant

1. **Install Dependencies**
   ```bash
   pip install qdrant-client
   ```

2. **Set up Qdrant Server**
   ```bash
   # Using Docker
   docker run -p 6333:6333 qdrant/qdrant
   
   # Or use Qdrant Cloud
   ```

3. **Update Code**
   ```python
   # Before
   from rag.nlp.search import Dealer
   dealer = Dealer(dataStore)
   
   # After
   from rag.nlp.qdrant_search import QdrantDealer
   from qdrant_client import QdrantClient
   
   client = QdrantClient(url="http://localhost:6333")
   dealer = QdrantDealer(client)
   ```

4. **Migrate Data**
   - Export vectors and metadata from existing system
   - Import into Qdrant collections using the same structure
   - Ensure vector dimensions match your embedding model

### Compatibility Notes

- All public methods maintain the same signature
- Return types and data structures are identical
- Error handling is consistent
- Logging follows the same patterns

## Performance Considerations

### Vector Search Optimization

- Use appropriate vector dimensions (384, 768, 1536, etc.)
- Consider using quantization for large datasets
- Implement proper indexing strategies

### Memory Usage

- Qdrant loads vectors into memory for fast access
- Monitor memory usage with large collections
- Use disk-based storage for very large datasets

### Scaling

- Qdrant supports horizontal scaling
- Consider sharding large collections
- Use replication for high availability

## Troubleshooting

### Common Issues

1. **Import Error: qdrant-client not found**
   ```bash
   pip install qdrant-client
   ```

2. **Connection Error: Cannot connect to Qdrant**
   - Verify Qdrant server is running
   - Check URL and port configuration
   - Verify network connectivity

3. **Dimension Mismatch**
   - Ensure vector dimensions match between embedding model and collection
   - Recreate collection with correct dimensions if needed

4. **Empty Results**
   - Check if collection exists and has data
   - Verify similarity threshold is not too high
   - Ensure proper filtering parameters

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When contributing to the Qdrant implementation:

1. Maintain interface compatibility with original Dealer
2. Add comprehensive type annotations
3. Include proper error handling and logging
4. Update tests and documentation
5. Follow existing code style and patterns

## License

This implementation follows the same Apache 2.0 license as the original RAGFlow project.