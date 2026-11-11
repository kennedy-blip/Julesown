import hmac
import hashlib
import os
import json
import logging
from typing import Optional
from fastapi import APIRouter, Request, Header, HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger("webhooks_router")
router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])

def verify_github_signature(payload_body: bytes, secret: str, signature_header: Optional[str]) -> bool:
    """
    Verifies that the webhook payload was sent from GitHub by validating SHA256 HMAC signature.
    Matches the X-Hub-Signature-256 header.
    """
    if not signature_header:
        return False

    if not signature_header.startswith("sha256="):
        return False

    expected_signature = signature_header[7:]
    computed_signature = hmac.new(
        key=secret.encode("utf-8"),
        msg=payload_body,
        digestmod=hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(computed_signature, expected_signature)


@router.post("/github")
async def github_webhook_endpoint(
    request: Request,
    x_hub_signature_256: Optional[str] = Header(None),
    x_github_event: Optional[str] = Header(None)
):
    """
    GitHub Webhook receiver. Verifies HMAC-SHA256 signature and handles repository events
    (e.g., issues, comments, pushes) to trigger Jules autonomous coding tasks.
    """
    webhook_secret = os.getenv("GITHUB_WEBHOOK_SECRET")
    body_bytes = await request.body()

    if webhook_secret:
        if not verify_github_signature(body_bytes, webhook_secret, x_hub_signature_256):
            logger.warning("Rejected webhook with invalid HMAC signature.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid X-Hub-Signature-256 HMAC signature"
            )
    else:
        logger.warning("GITHUB_WEBHOOK_SECRET is not configured; skipping HMAC verification.")

    try:
        payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Malformed JSON: {e}")

    event_type = x_github_event or "unknown"
    logger.info(f"Received GitHub webhook event: {event_type}")

    # Handle standard GitHub ping handshake
    if event_type == "ping":
        return {
            "status": "ok",
            "message": "GitHub ping received successfully",
            "zen": payload.get("zen")
        }

    # Dispatch tasks for issue / comment creation
    dispatched_task = None
    if event_type in ("issues", "issue_comment") and payload.get("action") in ("opened", "created"):
        issue = payload.get("issue", {})
        repo = payload.get("repository", {}).get("full_name", "unknown/repo")
        prompt = issue.get("body", "") or issue.get("title", "")

        dispatched_task = {
            "event": event_type,
            "repo": repo,
            "prompt": prompt,
            "status": "queued"
        }
        logger.info(f"Dispatched autonomous agent task for {repo}: {prompt}")

    return {
        "status": "received",
        "event": event_type,
        "dispatched": dispatched_task is not None,
        "task": dispatched_task
    }
