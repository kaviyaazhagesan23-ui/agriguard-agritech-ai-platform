import copy

import pytest

from src.matching import BuyerListing, FarmerListing, MatchingWeights, match_buyer, rank_buyers


def farmer(**overrides):
    values = dict(
        listing_id="F1", crop="Paddy", variety="B P T", grade="Local", quantity_quintals=100,
        farmer_market="Vallam", minimum_expected_price=2200, latitude=10.714, longitude=79.702,
        created_date="2026-09-21", synthetic_demo_data=True,
    )
    values.update(overrides)
    return FarmerListing(**values)


def buyer(**overrides):
    values = dict(
        buyer_id="B1", buyer_name="Synthetic Buyer", crop="Paddy", variety="B P T", grade="Local",
        required_quantity_quintals=80, offered_price_per_quintal=2400, preferred_market="Vallam",
        latitude=10.714, longitude=79.702, pickup_available=True, active=True, synthetic_demo_data=True,
    )
    values.update(overrides)
    return BuyerListing(**values)


def test_exact_crop_variety_grade_match():
    r = match_buyer(farmer(), buyer())
    assert r.eligible and r.crop_match and r.variety_match and r.grade_match


def test_crop_mismatch():
    r = match_buyer(farmer(), buyer(crop="Wheat"))
    assert not r.eligible and not r.crop_match


def test_variety_mismatch():
    r = match_buyer(farmer(), buyer(variety="1001"))
    assert r.eligible and not r.variety_match


def test_grade_mismatch():
    r = match_buyer(farmer(), buyer(grade="FAQ"))
    assert r.eligible and not r.grade_match


def test_inactive_buyer():
    r = match_buyer(farmer(), buyer(active=False))
    assert not r.eligible and "Buyer is inactive" in r.warnings


def test_insufficient_buyer_capacity():
    r = match_buyer(farmer(), buyer(required_quantity_quintals=101))
    assert not r.eligible and not r.quantity_compatible


def test_compatible_quantity():
    assert match_buyer(farmer(), buyer(required_quantity_quintals=100)).quantity_compatible


def test_offer_below_minimum():
    r = match_buyer(farmer(), buyer(offered_price_per_quintal=2199))
    assert not r.price_compatible


def test_offer_above_minimum():
    assert match_buyer(farmer(), buyer(offered_price_per_quintal=2200)).price_compatible


def test_pickup_available():
    assert match_buyer(farmer(), buyer(pickup_available=True)).pickup_available


def test_pickup_unavailable():
    r = match_buyer(farmer(), buyer(pickup_available=False))
    assert not r.pickup_available


def test_distance_calculation():
    r = match_buyer(farmer(), buyer(latitude=10.787, longitude=79.1378))
    assert r.distance_km is not None and r.distance_km > 0


def test_missing_coordinates():
    r = match_buyer(farmer(latitude=None, longitude=None), buyer())
    assert r.distance_km is None
    assert any("Distance unavailable" in w for w in r.warnings)


def test_deterministic_scoring():
    assert match_buyer(farmer(), buyer()).match_score == match_buyer(farmer(), buyer()).match_score


def test_deterministic_ranking():
    buyers = [buyer(buyer_id="B2", offered_price_per_quintal=2300), buyer(buyer_id="B1", offered_price_per_quintal=2300)]
    assert [r.buyer_id for r in rank_buyers(farmer(), buyers)] == ["B1", "B2"]


def test_tie_breaking_uses_price_then_id():
    results = rank_buyers(farmer(), [buyer(buyer_id="B2", offered_price_per_quintal=2300), buyer(buyer_id="B1", offered_price_per_quintal=2400)])
    assert results[0].buyer_id == "B1"


def test_invalid_farmer_quantity():
    with pytest.raises(ValueError):
        farmer(quantity_quintals=0)


def test_invalid_buyer_quantity():
    with pytest.raises(ValueError):
        buyer(required_quantity_quintals=0)


def test_invalid_price():
    with pytest.raises(ValueError):
        buyer(offered_price_per_quintal=-1)


def test_empty_buyer_list():
    assert rank_buyers(farmer(), []) == []


def test_multiple_buyer_ranking():
    results = rank_buyers(farmer(), [buyer(buyer_id="B2", offered_price_per_quintal=2300), buyer(buyer_id="B3", variety="1001"), buyer(buyer_id="B1", offered_price_per_quintal=2450)])
    assert results[0].buyer_id == "B1"
    assert all(r.eligible for r in results[:3])


def test_synthetic_data_flag_exists():
    assert farmer().synthetic_demo_data is True
    assert buyer().synthetic_demo_data is True
    assert match_buyer(farmer(), buyer()).synthetic_demo_data is True


def test_no_mutation_of_source_listings():
    f = farmer()
    b = buyer()
    f_before, b_before = copy.deepcopy(f), copy.deepcopy(b)
    match_buyer(f, b)
    assert f == f_before
    assert b == b_before


def test_same_inputs_same_results():
    f, b = farmer(), buyer()
    assert match_buyer(f, b) == match_buyer(f, b)


def test_weights_are_visible_and_configurable():
    weights = MatchingWeights(variety_match=50, grade_match=10, price_attractiveness=20, distance=10, pickup=10)
    r = match_buyer(farmer(), buyer(), weights)
    assert r.match_score >= 50
