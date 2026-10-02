from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

import jwt
from jwt import PyJWKClient

from apps.api.errors import Problem
from apps.api.models import Principal


class EntraTokens:
    def __init__(
        self,
        tenant_id: UUID,
        audience: UUID,
        key_resolver: Callable[[str], Any] | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.audience = str(audience)
        self.issuer = f"https://login.microsoftonline.com/{tenant_id}/v2.0"
        self.keys = PyJWKClient(
            f"https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys",
            cache_keys=True,
            lifespan=300,
            timeout=10,
        )
        self.key_resolver = key_resolver or (
            lambda token: self.keys.get_signing_key_from_jwt(token).key
        )

    def claims(self, authorization: str | None) -> dict:
        if not authorization or not authorization.startswith("Bearer "):
            raise Problem(
                401, "authentication_required", "An Entra bearer access token is required."
            )
        token = authorization[7:]
        if not token or len(token) > 16384:
            raise Problem(401, "invalid_token", "Invalid access token.")
        try:
            claims = jwt.decode(
                token,
                self.key_resolver(token),
                algorithms=["RS256"],
                audience=self.audience,
                issuer=self.issuer,
                options={"require": ["exp", "iat", "iss", "aud", "tid", "oid"]},
            )
        except jwt.PyJWKClientError as exc:
            raise Problem(
                503, "identity_unavailable", "Entra signing keys are unavailable."
            ) from exc
        except jwt.InvalidTokenError as exc:
            raise Problem(401, "invalid_token", "The access token is invalid or expired.") from exc
        if claims.get("tid") != str(self.tenant_id):
            raise Problem(403, "wrong_tenant", "The token belongs to a different tenant.")
        return claims

    def user(self, authorization: str | None) -> Principal:
        claims = self.claims(authorization)
        scopes = claims.get("scp")
        if not isinstance(scopes, str) or "access_as_user" not in scopes.split():
            raise Problem(403, "missing_scope", "Delegated access_as_user permission is required.")
        try:
            return Principal(tenant_id=self.tenant_id, object_id=UUID(claims["oid"]))
        except (ValueError, TypeError) as exc:
            raise Problem(
                401, "invalid_identity", "The token contains an invalid object ID."
            ) from exc

    def controller(self, authorization: str | None, allowed_principals: set[UUID]) -> None:
        claims = self.claims(authorization)
        try:
            principal_id = UUID(claims["oid"])
        except (ValueError, TypeError) as exc:
            raise Problem(401, "invalid_identity", "Invalid controller identity.") from exc
        if claims.get("scp") or principal_id not in allowed_principals:
            raise Problem(
                403,
                "controller_forbidden",
                "Only the configured managed identity may control this simulator.",
            )
