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


def compile_template(source, tmp_path):
    assert source.is_file()
    compiler = shutil.which("bicep")
    if compiler is None:
        installed = Path.home() / ".azure" / "bin" / "bicep"
        if installed.is_file():
            compiler = str(installed)
    if compiler is None:
        pytest.skip("Offline Bicep compiler not installed; source boundary test still runs.")
    output = tmp_path / f"{source.stem}.json"
    subprocess.run([compiler, "build", str(source), "--outfile", str(output)], check=True)
    return json.loads(output.read_text(encoding="utf-8"))


@pytest.fixture
def compiled(tmp_path):
    return compile_template(SOURCE, tmp_path)


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
    assert (
        scale["minReplicas"].replace(" ", "")
        == "[if(or(parameters('enabled'),parameters('artifactOpsEnabled')),1,0)]"
    )
    assert compiled["parameters"]["enabled"]["defaultValue"] is False
    assert compiled["parameters"]["artifactOpsEnabled"]["defaultValue"] is False
    assert compiled["parameters"]["artifactActorIds"]["defaultValue"] == []
    assert compiled["parameters"]["artifactMaxSeconds"]["maxValue"] == 1800
    assert compiled["parameters"]["artifactCaptureBytes"]["maxValue"] == 4 * 1024**3
    assert compiled["parameters"]["artifactDatasetBytes"]["maxValue"] == 20 * 1024**3


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
    assert compiled["parameters"]["reconciliationEnabled"]["defaultValue"] is False
    assert compiled["parameters"]["reconciliationActorIds"]["defaultValue"] == []
    assert compiled["parameters"]["reconciliationTargets"]["defaultValue"] == []
    for parameter, setting in (
        ("referenceCollectionsEnabled", "REFERENCE_COLLECTIONS_ENABLED"),
        ("pausedTrainingEnabled", "PAUSED_TRAINING_ENABLED"),
        ("pausedEvaluationEnabled", "PAUSED_EVALUATION_ENABLED"),
    ):
        assert compiled["parameters"][parameter]["defaultValue"] is False
        assert parameter in env[f"LEARNING_WORKER_{setting}"]
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
        "RECONCILIATION_ENABLED",
        "RECONCILIATION_ACTOR_IDS",
        "RECONCILIATION_TARGETS",
    ):
        assert f"LEARNING_WORKER_{field}" in env
    probes = container["probes"]
    assert {probe["type"] for probe in probes} == {"Startup", "Liveness", "Readiness"}
    assert all(probe["httpGet"] == {"path": "/healthz", "port": 8080} for probe in probes)


def test_optional_scheduler_is_off_scoped_bounded_and_has_no_extra_cloud_authority(tmp_path):
    source = ROOT / "infra" / "learning-reconciler.bicep"
    template = compile_template(source, tmp_path)
    resources = template["resources"]
    if isinstance(resources, dict):
        resources = list(resources.values())
    assert len(resources) == 1
    resource = resources[0]
    assert resource["type"] == "Microsoft.App/jobs"
    assert resource["condition"] == "[parameters('enabled')]"
    assert template["parameters"]["enabled"]["defaultValue"] is False
    assert template["parameters"]["reconciliationActorIds"]["defaultValue"] == []
    assert template["parameters"]["reconciliationTargets"]["defaultValue"] == []
    assert template["parameters"]["reconciliationTargets"]["maxLength"] == 20
    config = resource["properties"]["configuration"]
    assert config["triggerType"] == "Schedule"
    assert config["replicaTimeout"] == 120 and config["replicaRetryLimit"] == 0
    assert config["scheduleTriggerConfig"] == {
        "cronExpression": "* * * * *",
        "parallelism": 1,
        "replicaCompletionCount": 1,
    }
    container = resource["properties"]["template"]["containers"][0]
    assert container["command"][-1] == "apps.learning_worker.reconcile"
    assert "@sha256:" in container["image"]
    assert container["resources"] == {"cpu": "[json('0.5')]", "memory": "1Gi"}
    env = {entry["name"]: entry["value"] for entry in container["env"]}
    # ARM escapes a literal leading '[' rather than evaluating it as an expression.
    assert env["LEARNING_WORKER_ALLOWED_POLICY_TYPES"] == "[[]"
    assert env["LEARNING_WORKER_BOOTSTRAP_OWNER_IDS"] == "[[]"
    text = source.read_text(encoding="utf-8")
    for forbidden in (
        "roleAssignments@",
        "secretRef:",
        "ingress:",
        "Simulator.Control",
        "listKeys(",
    ):
        assert forbidden not in text
