import pytest

from learning.common import ContractError


@pytest.mark.parametrize(
    "family,revision",
    [
        ("gr00t_n1_5", "869830fc749c35f34771aa5209f923ac57e4564e"),
        ("gr00t_n1_6", "d0814e7ecb19202e7c8468b46098b0b7ef3a6d61"),
        ("gr00t_n1_7", "2fc962b973bccdd5d8ce4f67cc63b264d6886495"),
    ],
)
def test_known_noncommercial_or_conflicting_licenses_never_authorize_customer_use(family, revision):
    from learning.gr00t.licensing import require_commercial_model

    with pytest.raises(ContractError, match="license"):
        require_commercial_model(family, revision)


def test_an_unknown_revision_is_not_approved_by_default():
    from learning.gr00t.licensing import require_commercial_model

    with pytest.raises(ContractError, match="license"):
        require_commercial_model("gr00t_n1_7", "0" * 40)


def test_operator_checkbox_cannot_override_known_noncommercial_vendor_terms(tmp_path):
    from learning.gr00t.prepare import import_pretrained

    with pytest.raises(ContractError, match="license"):
        import_pretrained(
            tmp_path / "missing",
            tmp_path / "output",
            scope=None,
            profile=None,
            task=None,
            acknowledge_license_review=True,
        )
    assert not (tmp_path / "output").exists()
