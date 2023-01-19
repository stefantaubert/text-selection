from math import inf

from text_selection_core.filtering.vocabulary_filtering import get_match_oov_method


def test_1_inf__ab_b__true():
  method = get_match_oov_method(1, inf)
  assert method(["a", "b"], ["b"])


def test_1_inf__ab_ab__false():
  method = get_match_oov_method(1, inf)
  assert not method(["a", "b"], ["a", "b"])


def test_2_inf__aba_b__true():
  method = get_match_oov_method(2, inf)
  assert method(["a", "b", "a"], ["b"])


def test_2_inf__abb_b__false():
  method = get_match_oov_method(2, inf)
  assert not method(["a", "b", "b"], ["b"])


def test_2_3__aba_b__true():
  method = get_match_oov_method(2, 3)
  assert method(["a", "b", "a"], ["b"])


def test_2_3__abaa_b__false():
  method = get_match_oov_method(2, 3)
  assert not method(["a", "b", "a", "a"], ["b"])
