from math import inf

from text_selection_core.filtering.vocabulary_filtering import get_match_iv_method


def test_1_inf__ab_b__true():
  method = get_match_iv_method(1, inf)
  assert method(["a", "b"], ["b"])


def test_1_inf__ab_c__false():
  method = get_match_iv_method(1, inf)
  assert not method(["a", "b"], ["c"])


def test_2_inf__aba_a__true():
  method = get_match_iv_method(2, inf)
  assert method(["a", "b", "a"], ["a"])


def test_2_inf__abb_a__false():
  method = get_match_iv_method(2, inf)
  assert not method(["a", "b", "b"], ["a"])


def test_2_3__aba_b__true():
  method = get_match_iv_method(2, 3)
  assert method(["a", "b", "a"], ["a"])


def test_2_3__abaa_a__false():
  method = get_match_iv_method(2, 3)
  assert not method(["a", "b", "a", "a"], ["a"])
