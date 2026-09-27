from pathlib import Path


def test_policy_recipe_preserves_isaac_and_copies_a_real_isolated_python_installation():
    recipe = Path("simulation/Dockerfile.policy").read_text()
    assert "COPY --from=policy /usr/local /opt/lerobot-python" in recipe
    assert "COPY --from=policy /opt/smolvla-venv /opt/smolvla-venv" in recipe
    assert "COPY --from=policy /work /work" in recipe
    assert '"/opt/smolvla-venv/bin/$executable"' in recipe
    assert "/opt/lerobot-python/bin/python3.11" in recipe
    assert "libpython3.11.so.1.0" in recipe
    assert "COPY --from=policy /usr/local /usr/local" not in recipe
    assert "sys.version_info[:2] == (3, 11)" in recipe
    assert "sys.version_info[:2] == (3, 12)" in recipe
    assert "from learning.smolvla.checkpoint_runner import native_runtime" in recipe
    assert recipe.rstrip().endswith("USER 1234:1234")


def test_policy_recipe_requires_both_immutable_base_images():
    recipe = Path("simulation/Dockerfile.policy").read_text()
    assert 'case "$NATIVE_POLICY_IMAGE" in *@sha256:*)' in recipe
    assert 'case "$SIMULATOR_IMAGE" in *@sha256:*)' in recipe
    assert "ARG NATIVE_POLICY_IMAGE=" not in recipe
    assert "ARG SIMULATOR_IMAGE=" not in recipe
