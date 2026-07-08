from collections import Counter

from evaluation.gold_loader import REQUIRED_FIELDS, load_gold_set, validate_gold_set


def test_load_gold_set_loads_30_examples():
    examples = load_gold_set()
    assert len(examples) == 30


def test_all_ids_are_unique():
    examples = load_gold_set()
    ids = [example["id"] for example in examples]
    assert len(ids) == len(set(ids))
    assert ids[0] == "EVAL-001"
    assert ids[-1] == "EVAL-030"


def test_all_required_fields_exist():
    examples = load_gold_set()
    for example in examples:
        for field in REQUIRED_FIELDS:
            assert field in example, f"{example.get('id')} missing {field}"


def test_validate_gold_set_returns_no_errors_for_current_gold_set():
    examples = load_gold_set()
    assert validate_gold_set(examples) == []


def test_insufficient_examples_are_marked_and_have_empty_policy_ids():
    examples = load_gold_set()
    insufficient = [example for example in examples if example["id"] in {"EVAL-028", "EVAL-029", "EVAL-030"}]
    assert len(insufficient) == 3
    for example in insufficient:
        assert example["should_be_insufficient"] is True
        assert example["expected_policy_ids"] == []
        assert example["expected_policy_names"] == []
        assert example["expected_risk"] == ["Insufficient Evidence"]
        assert example["expected_category"] == ["Unknown or Insufficient Evidence"]


def test_at_least_3_insufficient_evidence_examples_exist():
    examples = load_gold_set()
    insufficient = [example for example in examples if example["should_be_insufficient"] is True]
    assert len(insufficient) >= 3


def test_category_distribution_matches_expected_id_ranges():
    examples = {example["id"]: example for example in load_gold_set()}

    expected_ranges = {
        "Insider": range(1, 6),
        "Whistleblowing": range(6, 11),
        "Related-Party": range(11, 16),
        "Derivative": range(16, 20),
        "FundsOrGuarantee": range(20, 24),
        "Asset": range(24, 28),
        "Unknown": range(28, 31),
    }

    assert {name: len(list(ids)) for name, ids in expected_ranges.items()} == {
        "Insider": 5,
        "Whistleblowing": 5,
        "Related-Party": 5,
        "Derivative": 4,
        "FundsOrGuarantee": 4,
        "Asset": 4,
        "Unknown": 3,
    }

    for index in expected_ranges["Insider"]:
        categories = " ".join(examples[f"EVAL-{index:03d}"]["expected_category"])
        assert "Insider Trading" in categories or "Material Information" in categories

    for index in expected_ranges["Whistleblowing"]:
        categories = " ".join(examples[f"EVAL-{index:03d}"]["expected_category"])
        assert "Whistleblowing" in categories or "Ethical Conduct" in categories

    for index in expected_ranges["Related-Party"]:
        categories = " ".join(examples[f"EVAL-{index:03d}"]["expected_category"])
        assert "Related-Party" in categories or "Asset Acquisition" in categories

    for index in expected_ranges["Derivative"]:
        categories = " ".join(examples[f"EVAL-{index:03d}"]["expected_category"])
        assert "Derivative" in categories

    for index in expected_ranges["FundsOrGuarantee"]:
        categories = " ".join(examples[f"EVAL-{index:03d}"]["expected_category"])
        assert "Funds Lending" in categories or "Endorsement and Guarantee" in categories

    for index in expected_ranges["Asset"]:
        categories = " ".join(examples[f"EVAL-{index:03d}"]["expected_category"])
        assert "Asset Acquisition" in categories

    for index in expected_ranges["Unknown"]:
        categories = " ".join(examples[f"EVAL-{index:03d}"]["expected_category"])
        assert "Unknown or Insufficient Evidence" in categories


def test_validate_gold_set_reports_duplicate_ids_readably():
    examples = load_gold_set()
    duplicated = examples + [dict(examples[2])]
    errors = validate_gold_set(duplicated)
    assert "Duplicate id found: EVAL-003" in errors


def test_validate_gold_set_reports_missing_field_readably():
    example = dict(load_gold_set()[11])
    example.pop("expected_risk")
    errors = validate_gold_set([example])
    assert "EVAL-012 missing required field: expected_risk" in errors


def test_validate_gold_set_reports_bad_insufficient_policy_ids_readably():
    example = dict(load_gold_set()[27])
    example["expected_policy_ids"] = ["WITS-001"]
    errors = validate_gold_set([example])
    assert "EVAL-028 insufficient example should have empty expected_policy_ids" in errors
