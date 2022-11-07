# from text_selection.common.ngram_extractor2 import get_ngrams_generator


# def test_n1_empty__returns_empty():
#   symbols = tuple()
#   result = list(get_ngrams_generator(symbols, n=1))
#   assert result == []


# def test_n2_empty__returns_empty():
#   symbols = tuple()
#   result = list(get_ngrams_generator(symbols, n=2))
#   assert result == []


# def test_n3_empty__returns_empty():
#   symbols = tuple()
#   result = list(get_ngrams_generator(symbols, n=3))
#   assert result == []


# def test_n2_a__returns_empty():
#   symbols = ("a",)
#   result = list(get_ngrams_generator(symbols, n=2))
#   assert result == []


# def test_n3_ab__returns_empty():
#   symbols = ("a", "b")
#   result = list(get_ngrams_generator(symbols, n=3))
#   assert result == []


# def test_n1_abc__returns_ab_bc():
#   symbols = ("a", "b", "c")
#   result = list(get_ngrams_generator(symbols, n=1))
#   assert result == [("a",), ("b",), ("c",)]


# def test_n2_abc__returns_ab_bc():
#   symbols = ("a", "b", "c")
#   result = list(get_ngrams_generator(symbols, n=2))
#   assert result == [("a", "b"), ("b", "c")]


# def test_n3_abc__returns_abc():
#   symbols = ("a", "b", "c")
#   result = list(get_ngrams_generator(symbols, n=3))
#   assert result == [("a", "b", "c")]
