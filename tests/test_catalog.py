import re

from conftest import CATALOG


def test_case_ids_and_requirement_links_are_complete():
    cases = CATALOG["cases"]
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
    requirements = set(CATALOG["requirements"])
    covered = set()
    gates = set()
    for case in cases:
        assert re.fullmatch(r"[A-Z][A-Z0-9]*-\d{3}", case["id"])
        assert case["title"].strip()
        assert case["requirement_ids"]
        assert set(case["requirement_ids"]) <= requirements
        covered.update(case["requirement_ids"])
        assert case["gate"] in CATALOG["required_release_gates"]
        gates.add(case["gate"])
    assert covered == requirements
    assert gates == set(CATALOG["required_release_gates"])


def test_implemented_cases_have_a_runner_and_planned_cases_are_not_claimed_as_passes():
    implemented = []
    for case in CATALOG["cases"]:
        assert case["status"] in {"implemented", "planned"}
        assert "passed" not in case
        if case["status"] == "implemented":
            implemented.append(case)
            assert case["gate"] == "G0"
            assert case["suite"] in {"environment", "finite", "cli"}
            if case["suite"] != "finite":
                assert isinstance(case["expected_valid"], bool)
                assert isinstance(case["expected_codes"], list)
            if case["suite"] == "cli":
                assert case["expected_exit"] == (0 if case["expected_valid"] else 2)
        else:
            for field in ("runner", "preconditions", "procedure"):
                assert case[field].strip()
            assert case["assertions"] and all(case["assertions"])
    assert implemented
