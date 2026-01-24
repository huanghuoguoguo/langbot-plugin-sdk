from __future__ import annotations

import abc
from typing import Any

from langbot_plugin.api.definition.components.base import PolymorphicComponent
from langbot_plugin.api.entities.builtin.rag.context import (
    RetrievalContext,
    RetrievalResultEntry,
    RetrievalResponse,
)
from langbot_plugin.api.entities.builtin.rag.models import (
    IngestionContext,
    IngestionResult,
)


class KnowledgeRetriever(PolymorphicComponent):
    """The knowledge retriever component.
    
    This is the legacy interface for knowledge retrieval.
    For new implementations, use RAGEngine instead.
    """

    __kind__ = "KnowledgeRetriever"

    @abc.abstractmethod
    async def retrieve(self, context: RetrievalContext) -> list[RetrievalResultEntry]:
        """Retrieve the data from the knowledge retriever.
        
        Args:
            context: The retrieval context.
            
        Returns:
            The retrieval result.
            The retrieval result is a list of RetrievalResultEntry.
            The RetrievalResultEntry contains the id, metadata, and distance of the retrieved data.
        """
        pass


class RAGEngine(PolymorphicComponent):
    """Complete RAG engine component with full lifecycle management.
    
    This component provides comprehensive RAG operations including document ingestion,
    deletion, and retrieval. It replaces the legacy KnowledgeRetriever with a more
    complete interface.
    
    Plugins implementing this component should:
    1. Handle document parsing and chunking in the ingest method
    2. Use host services (via context) for embedding and vector storage
    3. Implement lifecycle hooks for knowledge base creation/deletion
    4. Provide JSON schemas for creation and retrieval settings
    """

    __kind__ = "RAGEngine"

    # ========== Lifecycle Hooks ==========

    async def on_knowledge_base_create(self, kb_id: str, config: dict) -> None:
        """Called when a knowledge base using this engine is created.
        
        This is an optional hook for plugins to perform initialization tasks
        when a new knowledge base is created (e.g., setting up internal state,
        creating indices, etc.).
        
        Args:
            kb_id: The knowledge base identifier.
            config: Creation settings provided by the user.
        """
        pass

    async def on_knowledge_base_delete(self, kb_id: str) -> None:
        """Called when a knowledge base using this engine is deleted.
        
        This is an optional hook for plugins to perform cleanup tasks
        when a knowledge base is deleted (e.g., removing internal state,
        cleaning up resources, etc.).
        
        Note: The host will handle deletion of vectors from the vector store.
        
        Args:
            kb_id: The knowledge base identifier.
        """
        pass

    # ========== Core Methods ==========

    @abc.abstractmethod
    async def ingest(self, context: IngestionContext) -> IngestionResult:
        """Ingest a document into the knowledge base.
        
        This method should:
        1. Read the file from context.file_object.storage_path
        2. Parse the document content
        3. Chunk the content according to context.chunking_strategy
        4. Use host services to generate embeddings
        5. Use host services to store vectors
        6. Return the ingestion result
        
        Args:
            context: Ingestion context containing file info and settings.
            
        Returns:
            Ingestion result with status and metadata.
            
        Raises:
            IngestionError: If ingestion fails.
            ParsingError: If document parsing fails.
            ChunkingError: If chunking fails.
            HostServiceError: If host service calls fail.
        """
        pass

    @abc.abstractmethod
    async def delete_document(self, kb_id: str, document_id: str) -> bool:
        """Delete a document and its associated data from the knowledge base.
        
        This method should:
        1. Delete all chunks/vectors associated with the document
        2. Clean up any plugin-specific data structures
        
        Args:
            kb_id: Knowledge base identifier.
            document_id: Document identifier to delete.
            
        Returns:
            True if deletion was successful, False otherwise.
            
        Raises:
            HostServiceError: If vector deletion fails.
        """
        pass

    @abc.abstractmethod
    async def retrieve(self, context: RetrievalContext) -> RetrievalResponse:
        """Retrieve relevant content from the knowledge base.
        
        This method should:
        1. Process the query (optional: query rewriting, expansion)
        2. Use host services to generate query embedding
        3. Use host services to search vectors
        4. Optional: Apply reranking
        5. Return structured response
        
        Args:
            context: Retrieval context with query and settings.
            
        Returns:
            Structured retrieval response with results and metadata.
            
        Raises:
            RetrievalError: If retrieval fails.
            HostServiceError: If host service calls fail.
        """
        pass

    # ========== Schema Definitions ==========

    @abc.abstractmethod
    def get_creation_settings_schema(self) -> dict:
        """Get JSON Schema for knowledge base creation settings.
        
        This schema is used by the frontend to render a dynamic form when
        creating a knowledge base with this engine. The schema should follow
        JSON Schema Draft 7 specification.
        
        Example return value:
        {
            "type": "object",
            "properties": {
                "index_mode": {
                    "type": "string",
                    "enum": ["general", "qa", "parent_child"],
                    "default": "general",
                    "title": "Indexing Mode"
                },
                "chunk_size": {
                    "type": "integer",
                    "minimum": 100,
                    "maximum": 2000,
                    "default": 512,
                    "title": "Chunk Size"
                }
            },
            "required": ["index_mode"]
        }
        
        Returns:
            JSON Schema dict for creation settings.
        """
        pass

    @abc.abstractmethod
    def get_retrieval_settings_schema(self) -> dict:
        """Get JSON Schema for retrieval runtime settings.
        
        This schema is used by the frontend to render configuration options
        when using this knowledge base in a conversation or workflow.
        
        Example return value:
        {
            "type": "object",
            "properties": {
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 5,
                    "title": "Top K Results"
                },
                "similarity_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.7,
                    "title": "Similarity Threshold"
                },
                "enable_rerank": {
                    "type": "boolean",
                    "default": false,
                    "title": "Enable Reranking"
                }
            }
        }
        
        Returns:
            JSON Schema dict for retrieval settings.
        """
        pass
