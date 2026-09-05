# test_site_scorer.py
# Purpose: Automated tests for the site scorer.

import sys

sys.path.append("src/phase1_ml")

from site_scorer import score_location


# Test coordinate within the Agartala Municipal Corporation study area
TEST_LAT = 23.85
TEST_LON = 91.28


def test_score_returns_dict():
    result = score_location(TEST_LAT, TEST_LON)

    assert isinstance(result, dict)


def test_score_has_required_keys():
    result = score_location(TEST_LAT, TEST_LON)

    required_keys = [
        "latitude",
        "longitude",
        "opportunity_score",
        "risk_score",
        "features",
        "shap_explanation",
        "verdict"
    ]

    for key in required_keys:
        assert key in result


def test_scores_in_valid_range():
    result = score_location(TEST_LAT, TEST_LON)

    assert 0 <= result["opportunity_score"] <= 10
    assert 0 <= result["risk_score"] <= 10


def test_verdict_is_valid():
    result = score_location(TEST_LAT, TEST_LON)

    assert result["verdict"] in [
        "GOOD SITE",
        "POOR SITE"
    ]