from pathlib import Path

from scripts.check_docs import check_links, documents, markdown_targets


def test_repository_documentation_links_exist():
    root = Path(__file__).resolve().parents[1]
    assert check_links(root, documents(root)) == []


def test_link_parser_ignores_code_fences_and_collects_reference_links():
    text = """
[valid](../README.md#intro)
```markdown
[not a link](missing.md)
```
~~~python
[also not a link](absent.py)
~~~
[reference]: <guide%20name.md>
"""
    assert list(markdown_targets(text)) == ["../README.md#intro", "guide%20name.md"]


def test_links_reject_missing_and_outside_targets_but_allow_remote_and_anchors(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "guide name.md").write_text("# Guide\n", encoding="utf-8")
    readme = root / "README.md"
    readme.write_text(
        "[ok](guide%20name.md#heading)\n"
        "[web](https://example.com/absent)\n"
        "[mail](mailto:demo@example.com)\n"
        "[anchor](#heading)\n"
        "[missing](absent.md)\n"
        "[outside](../other.md)\n",
        encoding="utf-8",
    )
    assert check_links(root, [readme]) == [
        {"document": "README.md", "target": "absent.md", "reason": "missing_target"},
        {"document": "README.md", "target": "../other.md", "reason": "outside_repository"},
    ]
