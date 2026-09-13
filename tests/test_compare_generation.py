from scripts.compare_generation import common_prefix


def test_common_prefix_stops_at_first_divergence():
    assert common_prefix([1, 2, 3, 4], [1, 2, 9, 4]) == 2
