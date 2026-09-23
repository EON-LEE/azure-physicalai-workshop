import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "infra" / "learning-worker.bicep"


def test_private_worker_has_dedicated_identity_and_no_implicit_cloud_or_model_authority():
    assert SOURCE.is_file(), "A reproducible internal worker deployment template is required."
    text = SOURCE.read_text(encoding="utf-8")
    for field in (
        "param managedEnvironmentId string",
        "param workerIdentityResourceId string",
        "param apiPrincipalId string",
        "param enabled bool = false",
        "param allowedPolicyTypes array = []",
        "param bootstrapOwnerIds array = []",
        "name: 'LEARNING_WORKER_ALLOWED_API_PRINCIPALS'",
        "value: string([apiPrincipalId])",
    ):
        assert field in text
    assert "roleAssignments@" not in text
    assert "Simulator.Control" not in text
    assert "apiIdentityId" not in text
    assert "listKeys(" not in text
    assert "passwordSecretRef" not in text
    assert "secretRef:" not in text


@pytest.fixture
def compiled(tmp_path):
    assert SOURCE.is_file()
    compiler = shutil.which("bicep")
    if compiler is None:
        installed = Path.home() / ".azure" / "bin" / "bicep"
        if installed.is_file():
            compiler = str(installed)
    if compiler is None:
        pytest.skip("Offline Bicep compiler not installed; source boundary test still runs.")
    output = tmp_path / "learning-worker.json"
    subprocess.run([compiler, "build", str(SOURCE), "--outfile", str(output)], check=True)
    return json.loads(output.read_text(encoding="utf-8"))


def worker(template):
    resources = template["resources"]
    if isinstance(resources, dict):
        resources = list(resources.values())
    assert len(resources) == 1, (
        "This template must not create identities, RBAC, storage or GPU jobs."
    )
    assert resources[0]["type"] == "Microsoft.App/containerApps"
    return resources[0]


def test_compiled_worker_ingress_identity_and_scale_are_bounded(compiled):
    resource = worker(compiled)
    assert resource["identity"]["type"] == "UserAssigned"
    assert set(resource["identity"]["userAssignedIdentities"]) == {
        "[format('{0}', parameters('workerIdentityResourceId'))]"
    }
    properties = resource["properties"]
    assert properties["managedEnvironmentId"] == "[parameters('managedEnvironmentId')]"
    ingress = properties["configuration"]["ingress"]
    assert ingress["external"] is False
    assert ingress["allowInsecure"] is False
    assert ingress["targetPort"] == 8080
    assert properties["configuration"]["activeRevisionsMode"] == "Single"
    scale = properties["template"]["scale"]
    assert scale["maxReplicas"] == 1
    assert scale["minReplicas"].replace(" ", "") == "[if(parameters('enabled'),1,0)]"
    assert compiled["parameters"]["enabled"]["defaultValue"] is False


def test_compiled_image_and_managed_registry_are_digest_pinned_without_credentials(compiled):
    resource = worker(compiled)
    parameters = compiled["parameters"]
    assert "defaultValue" not in parameters["workerImageSha256"]
    assert parameters["workerImageSha256"]["minLength"] == 64
    assert parameters["workerImageSha256"]["maxLength"] == 64
    image = resource["properties"]["template"]["containers"][0]["image"]
    assert "@sha256:" in image
    assert "workerImageSha256" in image
    registry = resource["properties"]["configuration"]["registries"]
    assert registry == [
        {
            "server": "[parameters('registryServer')]",
            "identity": "[parameters('workerIdentityResourceId')]",
        }
    ]


def test_compiled_settings_have_single_api_caller_and_empty_model_bootstrap_defaults(compiled):
    container = worker(compiled)["properties"]["template"]["containers"][0]
    env = {entry["name"]: entry["value"] for entry in container["env"]}
    assert "apiPrincipalId" in env["LEARNING_WORKER_ALLOWED_API_PRINCIPALS"]
    assert "createArray" in env["LEARNING_WORKER_ALLOWED_API_PRINCIPALS"]
    assert compiled["parameters"]["allowedPolicyTypes"]["defaultValue"] == []
    assert compiled["parameters"]["bootstrapOwnerIds"]["defaultValue"] == []
    for field in (
        "TENANT_ID",
        "AUDIENCE",
        "MANAGED_IDENTITY_CLIENT_ID",
        "REGISTRY_ACCOUNT_URL",
        "REGISTRY_CONTAINER",
        "CAPTURE_ACCOUNT_URL",
        "CAPTURE_CONTAINER",
        "ALLOWED_POLICY_TYPES",
        "BOOTSTRAP_OWNER_IDS",
    ):
        assert f"LEARNING_WORKER_{field}" in env
    probes = container["probes"]
    assert {probe["type"] for probe in probes} == {"Startup", "Liveness", "Readiness"}
    assert all(probe["httpGet"] == {"path": "/healthz", "port": 8080} for probe in probes)
