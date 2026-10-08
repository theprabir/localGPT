import logging

from fastapi import APIRouter, Depends, HTTPException

from app.config import get_settings
from app.core.logging import get_logger
from app.models.database import get_database
from app.models.schemas import (
    ConversationCreateRequest,
    ConversationListResponse,
    ConversationResponse,
    ChatMessageResponse,
    MessageListResponse,
    AttachmentResponse,
)

logger = get_logger("localgpt.api.conversations")

router = APIRouter(prefix="/api", tags=["conversations"])


@router.get("/conversations", response_model=ConversationListResponse)
def list_conversations(
    limit: int = 50,
    offset: int = 0,
    db=Depends(get_database),
    settings=Depends(get_settings),
):
    if limit < 1:
        limit = 1
    if limit > 200:
        limit = 200
    conversations = db.list_conversations(limit=limit, offset=offset)
    return ConversationListResponse(
        conversations=[
            ConversationResponse(
                id=c["id"],
                title=c["title"],
                created_at=c["created_at"],
                updated_at=c["updated_at"],
                metadata=c.get("metadata"),
            )
            for c in conversations
        ]
    )


@router.post("/conversations", response_model=ConversationResponse)
def create_conversation(
    request: ConversationCreateRequest,
    db=Depends(get_database),
    settings=Depends(get_settings),
):
    if not request.title.strip():
        raise HTTPException(status_code=400, detail="Conversation title is required")
    conversation_id = db.create_conversation(title=request.title.strip())
    conv = db.get_conversation(conversation_id)
    assert conv is not None
    return ConversationResponse(
        id=conv["id"],
        title=conv["title"],
        created_at=conv["created_at"],
        updated_at=conv["updated_at"],
        metadata=conv.get("metadata"),
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
def get_conversation(
    conversation_id: int,
    db=Depends(get_database),
    settings=Depends(get_settings),
):
    conv = db.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationResponse(
        id=conv["id"],
        title=conv["title"],
        created_at=conv["created_at"],
        updated_at=conv["updated_at"],
        metadata=conv.get("metadata"),
    )


@router.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: int,
    db=Depends(get_database),
    settings=Depends(get_settings),
):
    deleted = db.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True}


@router.get("/conversations/{conversation_id}/messages", response_model=MessageListResponse)
def list_messages(
    conversation_id: int,
    db=Depends(get_database),
    settings=Depends(get_settings),
):
    conv = db.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    rows = db.list_messages(conversation_id)
    return MessageListResponse(
        messages=[
            ChatMessageResponse(
                id=m["id"],
                role=m["role"],
                content=m["content"],
                raw_content=m.get("raw_content"),
                attachment_id=m.get("attachment_id"),
                generated_image_id=m.get("generated_image_id"),
                created_at=m["created_at"],
            )
            for m in rows
        ]
    )


@router.get("/conversations/{conversation_id}/attachments")
def list_attachments(
    conversation_id: int,
    db=Depends(get_database),
    settings=Depends(get_settings),
):
    conv = db.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    rows = db.list_attachments_for_conversation(conversation_id)
    return [
        AttachmentResponse(
            id=a["id"],
            conversation_id=a["conversation_id"],
            original_filename=a["original_filename"],
            stored_filename=a["stored_filename"],
            media_type=a["media_type"],
            size_bytes=a["size_bytes"],
            width=a.get("width"),
            height=a.get("height"),
            created_at=a["created_at"],
        )
        for a in rows
    ]
