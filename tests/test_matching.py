"""Unit tests for tracking matching utilities."""
import pytest
np = pytest.importorskip("numpy")


def test_linear_assignment_empty_matrix():
    from lib.tracker.matching import linear_assignment
    cost = np.empty((0, 5))
    matches, unmatched_a, unmatched_b = linear_assignment(cost, thresh=0.5)
    assert matches.shape == (0, 2)
    assert len(unmatched_a) == 0
    assert len(unmatched_b) == 5


def test_linear_assignment_perfect_match():
    from lib.tracker.matching import linear_assignment
    # Identity cost matrix — perfect 1-to-1 match at cost 0
    cost = np.eye(3) * 0.1
    matches, unmatched_a, unmatched_b = linear_assignment(cost, thresh=0.5)
    assert len(matches) == 3
    assert len(unmatched_a) == 0
    assert len(unmatched_b) == 0


def test_linear_assignment_threshold_filters():
    from lib.tracker.matching import linear_assignment
    # All costs above threshold -> no matches
    cost = np.ones((2, 2)) * 0.9
    matches, unmatched_a, unmatched_b = linear_assignment(cost, thresh=0.5)
    assert len(matches) == 0
    assert len(unmatched_a) == 2
    assert len(unmatched_b) == 2


def test_iou_distance_empty():
    from lib.tracker.matching import iou_distance
    cost = iou_distance([], [])
    assert cost.shape == (0, 0)


def test_iou_distance_identical_boxes():
    from lib.tracker.matching import iou_distance
    box = np.array([[10.0, 20.0, 50.0, 60.0]])
    cost = iou_distance(box, box)
    assert cost.shape == (1, 1)
    np.testing.assert_almost_equal(cost[0, 0], 0.0, decimal=5)


def test_iou_distance_non_overlapping():
    from lib.tracker.matching import iou_distance
    box_a = np.array([[0.0, 0.0, 10.0, 10.0]])
    box_b = np.array([[100.0, 100.0, 200.0, 200.0]])
    cost = iou_distance(box_a, box_b)
    np.testing.assert_almost_equal(cost[0, 0], 1.0, decimal=5)


def test_merge_matches_basic():
    from lib.tracker.matching import merge_matches
    m1 = [(0, 1), (1, 2)]
    m2 = [(1, 0), (2, 1)]
    match, unmatched_O, unmatched_Q = merge_matches(m1, m2, shape=(2, 3, 2))
    # 0->1->0, 1->2->1
    match_set = set(match)
    assert (0, 0) in match_set
    assert (1, 1) in match_set
