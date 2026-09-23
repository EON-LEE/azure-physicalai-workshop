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
    learning_enabled: bool = False
    learning_policy_types: tuple[Literal["gr00t_n1_5", "gr00t_n1_7", "smolvla"], ...] = ()
    learning_bootstrap_principal_ids: frozenset[UUID] = frozenset()
    public_learning_owner_id: UUID | None = None
    public_learning_project_id: UUID | None = None
    public_learning_evaluation_id: UUID | None = None
    learning_worker_endpoint: str | None = None
    learning_worker_scope: str | None = Field(
        default=None, pattern=r"^api://[a-fA-F0-9-]{36}/\.default$"
    )
    learning_coach_agent_name: str | None = Field(default=None, min_length=1, max_length=128)
    learning_coach_agent_version: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("learning_worker_endpoint")
    @classmethod
    def private_learning_worker(cls, value):
        if value is None:
            return value
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
            or not parsed.hostname.endswith((".azurecontainerapps.io", ".azurewebsites.net"))
        ):
            raise ValueError("Configure an explicit HTTPS Azure managed learning worker origin.")
        return value.rstrip("/")

    @model_validator(mode="after")
    def complete_learning_endpoints(self):
        if bool(self.learning_worker_endpoint) != bool(self.learning_worker_scope):
            raise ValueError("Learning worker endpoint and managed identity scope must be paired.")
        if bool(self.learning_coach_agent_name) != bool(self.learning_coach_agent_version):
            raise ValueError("The separate Foundry learning coach requires a pinned version.")
        publication = (
            self.public_learning_owner_id,
            self.public_learning_project_id,
            self.public_learning_evaluation_id,
        )
        if any(publication) and not all(publication):
            raise ValueError("Public learning requires an explicit owner, project and evaluation.")
        return self

    public_demo_publish_live: bool = False
    public_demo_owner_id: UUID | None = None
    public_demo_environment_id: str | None = Field(
        default=None, pattern=r"^[a-z][a-z0-9-]*$", max_length=64
    )
    public_demo_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    public_demo_defect_environment_id: str | None = Field(
        default=None, pattern=r"^[a-z][a-z0-9-]*$", max_length=64
    )
    public_demo_defect_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    public_demo_presentation_id: str | None = Field(
        default=None, pattern=r"^[a-z][a-z0-9-]*$", max_length=64
    )
    public_demo_cases_presentation_id: str | None = Field(
        default=None, pattern=r"^[a-z][a-z0-9-]*$", max_length=64
    )

    @model_validator(mode="after")
    def publication_is_explicit(self):
        if self.public_demo_publish_live and not all(
            (self.public_demo_owner_id, self.public_demo_environment_id, self.public_demo_revision)
        ):
            raise ValueError(
                "Public live viewing requires an explicit owner, environment and revision."
            )
        paired = (
            self.public_demo_defect_environment_id,
            self.public_demo_defect_revision,
            self.public_demo_presentation_id,
        )
        if any(paired) and not (
            all(paired)
            and self.public_demo_publish_live
            and self.public_demo_environment_id != self.public_demo_defect_environment_id
            and self.public_demo_revision != self.public_demo_defect_revision
        ):
            raise ValueError("A presentation requires a complete, distinct pinned reference pair.")
        if self.public_demo_cases_presentation_id and not all(paired):
            raise ValueError("Recorded cases require an explicitly pinned reference pair.")
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
