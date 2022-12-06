import logging
import pickle
import random
import string
from functools import partial
from itertools import chain
from logging import getLogger
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from text_selection_core.types import Lines

chars = string.ascii_letters + string.punctuation


def random_string_generator(str_size: int, allowed_chars: str):
  return '|'.join(random.choice(allowed_chars) for _ in range(str_size))


def get_random_sentence(words_count: int, min_word_len: int, max_word_len: int, split: str) -> str:
  words = []
  for _ in range(words_count):
    words.append(random_string_generator(random.randint(min_word_len, max_word_len), chars))
  return f'{split} {split}'.join(words)


def get_random_sentences(lines_count: int, min_words_count: int, max_words_count: int, min_word_len: int, max_word_len: int, split: str) -> Lines:
  result = []
  for _ in range(lines_count):
    result.append(get_random_sentence(random.randint(min_words_count,
                  max_words_count), min_word_len, max_word_len, split))
  return result


def get_random_sentences_mp(lines_count: int, min_words_count: int, max_words_count: int, min_word_len: int, max_word_len: int, split: str) -> Lines:
  logger = getLogger(__name__)
  logger.info("Generating test sentences...")
  method = partial(
    get_random_sentence,
    min_word_len=min_word_len,
    max_word_len=max_word_len,
    split=split,
  )
  sentence_counts = (
    random.randint(min_words_count, max_words_count)
    for _ in range(lines_count)
  )

  with Pool(processes=cpu_count(), maxtasksperchild=None) as pool:
    sentences = list(tqdm(pool.imap_unordered(
        method, sentence_counts, chunksize=4000
    ), total=lines_count))

  logger.info("Done generating")
  return sentences


def get_random_sentences_mp2(lines_count_per_job: int, jobs: int, min_words_count: int, max_words_count: int, min_word_len: int, max_word_len: int, split: str) -> Lines:
  logger = getLogger(__name__)
  logger.info("Generating test sentences...")
  method = partial(
    get_random_sentences,
    min_words_count=min_words_count,
    max_words_count=max_words_count,
    min_word_len=min_word_len,
    max_word_len=max_word_len,
    split=split,
  )

  sentence_counts = [
    lines_count_per_job
    for j in range(jobs)
  ]

  with Pool(processes=cpu_count(), maxtasksperchild=None) as pool:
    sentences = list(tqdm(pool.imap_unordered(
        method, sentence_counts, chunksize=1
    ), total=len(sentence_counts), desc="Generating data"))
  logger.info("Done generating")
  sentences = list(chain(*sentences))
  logger.info("Done chaining")
  return sentences


# def test_stress_test():
#   logger = getLogger()

#   lines = get_random_sentences_mp(1920000, 3, 10, 3, 7, "|")
#   count_per_job = 20_000
#   jobs = cpu_count() * 6
#   #line_nrs = OrderedSet(tqdm(list(range(count)), desc="Building index", unit=TQDM_LINE_UNIT))
#   line_nrs = range(count_per_job)
#   logger.info(f"Generating {count_per_job * jobs} lines.")
#   lines = get_random_sentences_mp2(count_per_job, jobs, 3, 10, 3, 7, "|")

#   logger.info(f"Generated {len(lines)} lines.")
#   # lines = list(tqdm(get_random_sentences(count, 3, 10, 3, 7, "|"),
#   #              total=count, desc="Generating data", unit=TQDM_LINE_UNIT))
#   # lines = [
#   #   f"{i}|a|b|c|{i}"
#   #   for i in tqdm(range(count), desc="Generating data", unit=TQDM_LINE_UNIT)
#   # ]

#   #line_nrs = range(0, count, 2)


def generate_small_set():
  lines = get_random_sentences_mp(1_000_000, 3, 10, 3, 7, "|")

  main_logger = getLogger()
  main_logger.info("Saving")
  with open("/tmp/test_data_small.pkl", mode="wb") as f:
    pickle.dump(lines, f)


def load_small_test_set() -> Lines:
  main_logger = getLogger()
  main_logger.info("Loading")
  with open("/tmp/test_data_small.pkl", mode="rb") as f:
    lines = pickle.load(f)
  return lines


def generate_big_set():
  lines = get_random_sentences_mp(40_000_000, 3, 10, 3, 7, "|")

  main_logger = getLogger()
  main_logger.info("Saving")
  with open("/tmp/test_data_big.pkl", mode="wb") as f:
    pickle.dump(lines, f)


def load_big_test_set() -> Lines:
  main_logger = getLogger()
  main_logger.info("Loading")
  with open("/tmp/test_data_big.pkl", mode="rb") as f:
    lines = pickle.load(f)
  return lines


if __name__ == "__main__":
  # NOTE: start without debugging for speed!
  main_logger = getLogger()
  main_logger.setLevel(logging.DEBUG)
  main_logger.manager.disable = logging.NOTSET
  # generate_small_set()
  # load_small_test_set()
  generate_big_set()
  # load_big_test_set()
