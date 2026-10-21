from uuid import UUID, uuid4

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.backend.auth.dependencies import get_current_user
from src.backend.database.connection import DATABASE_URL
from fastapi import Request
from langchain_core.messages import AIMessage, HumanMessage


router = APIRouter(
    prefix="/api/v1/conversations",
    tags=["Conversations"],
)


@router.post("", status_code=status.HTTP_201_CREATED)
def create_conversation(
    current_user: dict = Depends(get_current_user),
):
    """Create an empty conversation owned by the authenticated user."""
    conversation_id = uuid4()
    user_id = current_user["user_id"]

    try:
        with psycopg.connect(
            DATABASE_URL,
            connect_timeout=5,
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO conversations (conversation_id, user_id)
                    VALUES (%s, %s)
                    RETURNING conversation_id, created_at, updated_at
                    """,
                    (conversation_id, user_id),
                )
                row = cursor.fetchone()

        return {
            "conversation_id": str(row[0]),
            "created_at": row[1],
            "updated_at": row[2],
        }

    except psycopg.Error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation service is temporarily unavailable",
        )


@router.get("")
def list_conversations(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: dict = Depends(get_current_user),
):
    """List only the authenticated user's conversations."""
    user_id = current_user["user_id"]

    try:
        with psycopg.connect(
            DATABASE_URL,
            connect_timeout=5,
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT conversation_id, created_at, updated_at
                    FROM conversations
                    WHERE user_id = %s
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (user_id, limit, offset),
                )

                rows = cursor.fetchall()

                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM conversations
                    WHERE user_id = %s
                    """,
                    (user_id,),
                )
                total = cursor.fetchone()[0]

        return {
            "items": [
                {
                    "conversation_id": str(row[0]),
                    "created_at": row[1],
                    "updated_at": row[2],
                }
                for row in rows
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    except psycopg.Error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation service is temporarily unavailable",
        )
@router.get("/{conversation_id}/messages")
def get_conversation_messages(
    conversation_id: UUID,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Retrieve checkpointed user/assistant messages for an owned conversation."""
    user_id = str(current_user["user_id"])

    # Verify ownership before reading checkpoint data.
    try:
        with psycopg.connect(
            DATABASE_URL,
            connect_timeout=5,
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT 1
                    FROM conversations
                    WHERE conversation_id = %s
                      AND user_id = %s
                    """,
                    (conversation_id, user_id),
                )
                owned = cursor.fetchone()

    except psycopg.Error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation service is temporarily unavailable",
        )

    if not owned:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    graph = getattr(request.app.state, "chat_graph", None)

    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not initialized",
        )

    thread_id = f"{user_id}:{conversation_id}"

    try:
        snapshot = graph.get_state(
            config={
                "configurable": {
                    "thread_id": thread_id,
                }
            }
        )

        values = snapshot.values or {}
        messages = []

        for message in values.get("messages", []):
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                continue

            content = message.content

            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                text = "\n".join(
                    item["text"]
                    for item in content
                    if isinstance(item, dict)
                    and isinstance(item.get("text"), str)
                ).strip()
            else:
                text = ""

            if text:
                messages.append({
                    "role": role,
                    "content": text,
                })

        return {
            "conversation_id": str(conversation_id),
            "messages": messages,
            "count": len(messages),
        }

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to retrieve conversation history",
        )