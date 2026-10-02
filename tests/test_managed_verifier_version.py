from apps.learning_worker.managed_reports import _version


def test_verifier_hashes_source_and_dependency_lock_not_generated_environments(tmp_path):
    worker = tmp_path / "apps" / "learning_worker"
    worker.mkdir(parents=True)
    source = worker / "managed_reports.py"
    source.write_text("source-v1")
    lock = worker / "uv.lock"
    lock.write_text("locked-dependencies-v1")
    before = _version(tmp_path)
    for relative in (
        ".venv/site-packages/sdk.py",
        "__pycache__/generated.py",
        "node_modules/generated.json",
        ".cache/downloaded.json",
    ):
        generated = worker / relative
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text("not-verifier-source")
    assert _version(tmp_path) == before
    source.write_text("source-v2")
    changed_source = _version(tmp_path)
    assert changed_source != before
    lock.write_text("locked-dependencies-v2")
    assert _version(tmp_path) != changed_source
