"""Unit tests for tracking utility functions."""
import os
import tempfile
import pytest
np = pytest.importorskip("numpy")
from lib.tracking_utils.utils import mkdir_if_missing
from lib.tracking_utils.timer import Timer


def test_mkdir_if_missing_creates_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = os.path.join(tmpdir, 'new_subdir')
        assert not os.path.exists(new_dir)
        mkdir_if_missing(new_dir)
        assert os.path.isdir(new_dir)


def test_mkdir_if_missing_existing_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Should not raise when dir already exists
        mkdir_if_missing(tmpdir)
        assert os.path.isdir(tmpdir)


def test_timer_tic_toc():
    timer = Timer()
    timer.tic()
    timer.toc()
    assert timer.total_time > 0
    assert timer.calls == 1
    assert timer.average_time > 0


def test_timer_multiple_calls():
    timer = Timer()
    for _ in range(3):
        timer.tic()
        timer.toc()
    assert timer.calls == 3
    assert timer.average_time == pytest.approx(timer.total_time / 3, rel=1e-5)


def test_timer_reset():
    timer = Timer()
    timer.tic()
    timer.toc()
    timer.clear()
    assert timer.calls == 0
    assert timer.total_time == 0
