from streamguard_bench.data.token_coordinates import _first_overlapping_token


def test_boundary_inside_token_maps_to_first_intersecting_token():
    offsets = ((0, 2), (2, 5), (5, 8))
    assert _first_overlapping_token(offsets, 3) == 1
    assert _first_overlapping_token(offsets, 5) == 2
    assert _first_overlapping_token(offsets, 8) == 3
