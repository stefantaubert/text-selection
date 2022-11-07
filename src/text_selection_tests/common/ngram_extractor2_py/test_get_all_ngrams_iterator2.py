# from ordered_set import OrderedSet
# from text_selection.common.ngram_extractor2 import get_all_ngrams_iterator


# def test_unigram_empty__returns_no_unigram():
#   result = get_all_ngrams_iterator(OrderedSet(), n_gram=1)
#   assert list(result) == []


# def test_unigram_a__returns_a():
#   result = get_all_ngrams_iterator(OrderedSet(("a",)), n_gram=1)
#   assert list(result) == [("a",)]


# def test_unigram_a_b__returns_a_b():
#   result = get_all_ngrams_iterator(OrderedSet(("a", "b")), n_gram=1)
#   assert list(result) == [("a",), ("b",)]


# def test_bigram_empty__returns_no_bigram():
#   result = get_all_ngrams_iterator(OrderedSet(), n_gram=2)
#   assert list(result) == []


# def test_bigram_a__returns__aa():
#   result = get_all_ngrams_iterator(OrderedSet(("a",)), n_gram=2)
#   assert list(result) == [("a", "a")]


# def test_bigram_ab__returns_aa_ab_ba_bb():
#   result = get_all_ngrams_iterator(OrderedSet(("a", "b")), n_gram=2)
#   assert list(result) == [("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]


# def test_trigram_empty__returns_no_trigram():
#   result = get_all_ngrams_iterator(OrderedSet(), n_gram=3)
#   assert list(result) == []


# def test_trigram_a__returns__aaa():
#   result = get_all_ngrams_iterator(OrderedSet(("a",)), n_gram=2)
#   assert list(result) == [("a", "a", "a")]


# def test_trigram_ab__returns_all_8_trigrams():
#   # 8 = 2*2*2
#   result = list(get_all_ngrams_iterator(OrderedSet(("a", "b")), n_gram=3))
#   assert result == [
#     ('a', 'a', 'a'),
#     ('a', 'a', 'b'),
#     ('a', 'b', 'a'),
#     ('a', 'b', 'b'),
#     ('b', 'a', 'a'),
#     ('b', 'a', 'b'),
#     ('b', 'b', 'a'),
#     ('b', 'b', 'b'),
#   ]


# def test_trigram_abc__returns_all_27_trigrams():
#   # 27 = 3*3*3
#   result = list(get_all_ngrams_iterator(OrderedSet(("a", "b", "c")), n_gram=3))
#   assert list(result) == [
#     ('a', 'a', 'a'),
#     ('a', 'a', 'b'),
#     ('a', 'a', 'c'),
#     ('a', 'b', 'a'),
#     ('a', 'b', 'b'),
#     ('a', 'b', 'c'),
#     ('a', 'c', 'a'),
#     ('a', 'c', 'b'),
#     ('a', 'c', 'c'),
#     ('b', 'a', 'a'),
#     ('b', 'a', 'b'),
#     ('b', 'a', 'c'),
#     ('b', 'b', 'a'),
#     ('b', 'b', 'b'),
#     ('b', 'b', 'c'),
#     ('b', 'c', 'a'),
#     ('b', 'c', 'b'),
#     ('b', 'c', 'c'),
#     ('c', 'a', 'a'),
#     ('c', 'a', 'b'),
#     ('c', 'a', 'c'),
#     ('c', 'b', 'a'),
#     ('c', 'b', 'b'),
#     ('c', 'b', 'c'),
#     ('c', 'c', 'a'),
#     ('c', 'c', 'b'),
#     ('c', 'c', 'c'),
#   ]
