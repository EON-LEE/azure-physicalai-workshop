"""Shared request limits without importing the web application's data or startup logic."""

from fastapi.responses import JSONResponse


class BodyLimit:
    def __init__(self, app, limit: int = 1024 * 1024) -> None:
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.limit:
                response = JSONResponse(
                    {
                        "error": {
                            "code": "body_too_large",
                            "message": "Request exceeds 1 MiB.",
                            "details": [],
                        }
                    },
                    status_code=413,
                )
                return await response(scope, receive, send)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
