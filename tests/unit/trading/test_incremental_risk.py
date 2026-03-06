import pytest

from src.trading.risk_kernels import IncrementalDeltaTracker


def test_incremental_delta_tracker_success():
    tracker = IncrementalDeltaTracker(initial_delta=0.0, max_net_delta=100.0)
    
    # Trade 1: OK
    assert tracker.validate_and_update(50.0) is True
    assert tracker.current_net_delta == 50.0
    
    # Trade 2: OK
    assert tracker.validate_and_update(40.0) is True
    assert tracker.current_net_delta == 90.0
    
    # Trade 3: OK (Negative delta)
    assert tracker.validate_and_update(-20.0) is True
    assert tracker.current_net_delta == 70.0

def test_incremental_delta_tracker_veto():
    tracker = IncrementalDeltaTracker(initial_delta=90.0, max_net_delta=100.0)
    
    # Trade 1: Veto (exceeds 100)
    assert tracker.validate_and_update(15.0) is False
    assert tracker.current_net_delta == 90.0 # No change
    
    # Trade 2: OK (Stay at limit)
    assert tracker.validate_and_update(10.0) is True
    assert tracker.current_net_delta == 100.0

def test_incremental_delta_tracker_negative_limit():
    tracker = IncrementalDeltaTracker(initial_delta=-90.0, max_net_delta=100.0)
    
    # Trade 1: Veto (exceeds -100)
    assert tracker.validate_and_update(-15.0) is False
    assert tracker.current_net_delta == -90.0
    
    # Trade 2: OK
    assert tracker.validate_and_update(-10.0) is True
    assert tracker.current_net_delta == -100.0

def test_incremental_delta_tracker_reset():
    tracker = IncrementalDeltaTracker(initial_delta=50.0)
    tracker.reset(10.0)
    assert tracker.current_net_delta == 10.0
