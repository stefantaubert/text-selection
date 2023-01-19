from text_selection_core.filtering.vocabulary_filtering import matches_any_oov


def test_empty__returns_false():
  assert not matches_any_oov([], ["a"])


def test_ab_b__returns_true():
  assert matches_any_oov(["a", "b"], ["b"])


def test_ab_ab__returns_false():
  assert not matches_any_oov(["a", "b"], ["a", "b"])
