"""Host services protocol definitions for RAG plugins."""

from __future__ import annotations

import abc
from typing import Any, BinaryIO, Protocol

from .models import FileStreamHandle


class EmbedderProtocol(Protocol):
    """Protocol for embedding generation provided by host."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        ...


class VectorStoreProtocol(Protocol):
    """Protocol for vector store operations provided by host."""

    async def upsert(
        self,
        collection_id: str,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Insert or update vectors in the collection.

        Args:
            collection_id: Target collection identifier.
            ids: List of vector IDs.
            vectors: List of embedding vectors.
            metadata: Optional metadata for each vector.

        Raises:
            VectorStoreError: If upsert operation fails.
            CollectionNotFoundError: If collection is not accessible.
        """
        ...

    async def search(
        self,
        collection_id: str,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors.

        Args:
            collection_id: Target collection identifier.
            query_vector: Query embedding vector.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of search results with 'id', 'score', and 'metadata'.

        Raises:
            VectorStoreError: If search operation fails.
            CollectionNotFoundError: If collection is not accessible.
        """
        ...

    async def delete(
        self,
        collection_id: str,
        ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> int:
        """Delete vectors from the collection.

        Args:
            collection_id: Target collection identifier.
            ids: Optional list of vector IDs to delete.
            filters: Optional metadata filters for deletion.

        Returns:
            Number of vectors deleted.

        Raises:
            VectorStoreError: If delete operation fails.
            CollectionNotFoundError: If collection is not accessible.
        """
        ...

    async def count(
        self,
        collection_id: str,
        filters: dict[str, Any] | None = None,
    ) -> int:
        """Count vectors in the collection.

        Args:
            collection_id: Target collection identifier.
            filters: Optional metadata filters.

        Returns:
            Number of vectors matching the criteria.

        Raises:
            VectorStoreError: If count operation fails.
            CollectionNotFoundError: If collection is not accessible.
        """
        ...


class HostServices(abc.ABC):
    """Abstract base class for host services provided to RAG plugins.

    This class defines the interface through which RAG plugins can access
    host-provided capabilities like embedding generation and vector storage.
    Each plugin instance receives a scoped HostServices instance that only
    allows access to its designated collection.
    """

    @property
    @abc.abstractmethod
    def embedder(self) -> EmbedderProtocol:
        """Get the embedder service.

        Returns:
            Embedder protocol implementation.
        """
        ...

    @property
    @abc.abstractmethod
    def vector_store(self) -> VectorStoreProtocol:
        """Get the vector store service.

        Returns:
            Vector store protocol implementation.
        """
        ...

    @property
    @abc.abstractmethod
    def collection_id(self) -> str:
        """Get the collection ID assigned to this plugin instance.

        Returns:
            Collection identifier.
        """
        ...

    @abc.abstractmethod
    async def get_file_stream(self, storage_path: str) -> tuple[BinaryIO, FileStreamHandle]:
        """Get a file stream for reading.

        Args:
            storage_path: Path to the file in storage system.

        Returns:
            Tuple of (file stream, handle for cleanup).

        Raises:
            FileServiceError: If file cannot be opened.
        """
        ...

    @abc.abstractmethod
    async def close_file_stream(self, handle: FileStreamHandle) -> None:
        """Close a file stream.

        Args:
            handle: Handle returned from get_file_stream.

        Raises:
            FileServiceError: If stream cannot be closed.
        """
        ...
