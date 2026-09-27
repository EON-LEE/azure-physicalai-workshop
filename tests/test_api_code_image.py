from pathlib import Path


def test_api_code_overlay_preserves_runtime_and_removes_stale_web_assets():
    recipe = Path("Dockerfile.api-code").read_text()
    assert "ARG API_BASE_IMAGE=" not in recipe
    assert 'case "$API_BASE_IMAGE" in *@sha256:*)' in recipe
    assert "shutil.rmtree('/app/apps/web/dist')" in recipe
    assert "COPY --chown=10001:10001 apps/api/ apps/api/" in recipe
    assert "COPY --chown=10001:10001 apps/web/dist/ apps/web/dist/" in recipe
    assert "ENTRYPOINT" not in recipe and "CMD" not in recipe
    assert "pip install" not in recipe and "uv sync" not in recipe
    assert recipe.rstrip().endswith("USER 10001:10001")
