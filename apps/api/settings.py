from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    deployment_mode: Literal["azure"] = "azure"
    entra_tenant_id: UUID
    entra_spa_client_id: UUID
    entra_api_client_id: UUID
    azure_client_id: UUID
    foundry_project_endpoint: str
    foundry_agent_name: str = Field(min_length=1)
    foundry_agent_version: str | None = Field(default=None, min_length=1)
    cosmos_endpoint: str
    cosmos_database: str = "physicalai"
    cosmos_container: str = "state"
    storage_account_url: str
    storage_container: str = "artifacts"
    sim_bridge_endpoint: str
    sim_bridge_ca_pem: str | None = None
    runtime_bootstrap_blob: str = "bootstrap/runtime.json"
    sim_ca_blob: str = "bootstrap/simulator-ca.pem"
    applicationinsights_connection_string: str | None = None
    web_dist: Path = Path("apps/web/dist")
    model_timeout_seconds: float = Field(default=40, gt=0, le=120)
    bridge_timeout_seconds: float = Field(default=10, gt=0, le=30)
    approval_ttl_seconds: int = Field(default=300, ge=5, le=3600)
    public_demo_publish_live: bool = False
    public_demo_owner_id: UUID | None = None
    public_demo_environment_id: str | None = Field(
        default=None, pattern=r"^[a-z][a-z0-9-]*$", max_length=64
    )
    public_demo_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def publication_is_explicit(self):
        if self.public_demo_publish_live and not all(
            (self.public_demo_owner_id, self.public_demo_environment_id, self.public_demo_revision)
        ):
            raise ValueError(
                "Public live viewing requires an explicit owner, environment and revision."
            )
        return self

    @field_validator(
        "foundry_project_endpoint", "cosmos_endpoint", "storage_account_url", "sim_bridge_endpoint"
    )
    @classmethod
    def https_endpoint(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "An HTTPS service endpoint without credentials/query/fragment is required."
            )
        return value.rstrip("/")

    @field_validator("foundry_project_endpoint", "cosmos_endpoint", "storage_account_url")
    @classmethod
    def azure_endpoint(cls, value: str, info: ValidationInfo) -> str:
        parsed = urlsplit(value)
        suffix = {
            "foundry_project_endpoint": ".ai.azure.com",
            "cosmos_endpoint": ".documents.azure.com",
            "storage_account_url": ".blob.core.windows.net",
        }[info.field_name]
        if not parsed.hostname or not parsed.hostname.endswith(suffix):
            raise ValueError(
                "This runtime requires the documented Azure public-cloud service endpoint."
            )
        if info.field_name == "foundry_project_endpoint" and not parsed.path.startswith(
            "/api/projects/"
        ):
            raise ValueError(
                "A Foundry project endpoint is required, not a model or third-party API URL."
            )
        return value

    @property
    def delegated_scope(self) -> str:
        return f"api://{self.entra_api_client_id}/access_as_user"

    @property
    def simulator_scope(self) -> str:
        return f"api://{self.entra_api_client_id}/.default"
