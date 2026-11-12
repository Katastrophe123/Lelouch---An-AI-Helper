"""Your knowledge base: upload, list, delete and search documents."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, File, Path, Response, UploadFile, status

from app.api.deps import CurrentUserDep, KnowledgeServiceDep, SettingsDep
from app.api.errors import (
    BadRequestError,
    NotFoundError,
    PayloadTooLargeError,
    ServiceUnavailableError,
    UnsupportedMediaTypeError,
)
from app.knowledge.embeddings import EmbedderUnavailableError
from app.knowledge.parsing import EmptyDocumentError, UnsupportedDocumentError
from app.knowledge.service import DocumentNotFoundError, DocumentTooLargeError
from app.models.schemas import DocumentResponse, DocumentSearchHit, DocumentSearchRequest

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document (txt, md, pdf, docx, csv, code...)",
    responses={
        400: {"description": "The file has no readable text"},
        413: {"description": "The file is larger than MAX_UPLOAD_MB"},
        415: {"description": "Unsupported file type"},
        503: {"description": "The embedding model could not be loaded"},
    },
)
async def upload_document(
    user: CurrentUserDep,
    knowledge: KnowledgeServiceDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="The document to add")],
) -> DocumentResponse:
    # Read one byte past the limit so an oversized upload is rejected without
    # buffering the whole thing.
    data = await file.read(settings.max_upload_bytes + 1)
    filename = file.filename or "upload.txt"
    try:
        document = await knowledge.ingest(
            user_id=user.id, filename=filename, content_type=file.content_type, data=data
        )
    except DocumentTooLargeError as exc:
        raise PayloadTooLargeError(str(exc)) from exc
    except UnsupportedDocumentError as exc:
        raise UnsupportedMediaTypeError(str(exc)) from exc
    except EmptyDocumentError as exc:
        raise BadRequestError(str(exc)) from exc
    except EmbedderUnavailableError as exc:
        raise ServiceUnavailableError(str(exc)) from exc
    return DocumentResponse.model_validate(document)


@router.get("", response_model=list[DocumentResponse], summary="List your documents")
async def list_documents(
    user: CurrentUserDep, knowledge: KnowledgeServiceDep
) -> list[DocumentResponse]:
    documents = await knowledge.list_documents(user_id=user.id)
    return [DocumentResponse.model_validate(doc) for doc in documents]


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its embeddings",
    responses={404: {"description": "Document not found or not owned by the caller"}},
)
async def delete_document(
    user: CurrentUserDep,
    knowledge: KnowledgeServiceDep,
    document_id: Annotated[uuid.UUID, Path(description="Document identifier")],
) -> Response:
    try:
        await knowledge.delete(document_id, user_id=user.id)
    except DocumentNotFoundError as exc:
        raise NotFoundError(str(exc), details={"document_id": str(document_id)}) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/search",
    response_model=list[DocumentSearchHit],
    summary="Search your documents (the same search the agent uses)",
)
async def search_documents(
    payload: DocumentSearchRequest, user: CurrentUserDep, knowledge: KnowledgeServiceDep
) -> list[DocumentSearchHit]:
    try:
        hits = await knowledge.search(user_id=user.id, query=payload.query, top_k=payload.top_k)
    except EmbedderUnavailableError as exc:
        raise ServiceUnavailableError(str(exc)) from exc
    return [
        DocumentSearchHit(
            document_id=hit.document_id,
            filename=hit.filename,
            chunk_index=hit.chunk_index,
            score=hit.score,
            content=hit.content,
        )
        for hit in hits
    ]
