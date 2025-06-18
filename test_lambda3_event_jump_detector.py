import numpy as np
from event_jump_detector import generate_data_pattern, calc_lambda3_features_v2

def test_generate_data_pattern():
    # Test "single_jump" pattern, T=100
    data, trend, jumps = generate_data_pattern(pattern="single_jump", T=100)
    assert len(data) == 100
    assert np.sum(jumps != 0) == 1

def test_calc_lambda3_features_v2():
    data, trend, jumps = generate_data_pattern(pattern="multi_jump", T=150)
    pos, neg, rho_T, time_trend = calc_lambda3_features_v2(data)
    assert len(pos) == 150
    assert len(neg) == 150
    assert len(rho_T) == 150
    assert len(time_trend) == 150
