import os
from random import uniform
from typing import Dict, List, OrderedDict

import numpy as np
import pandas as pd
from scipy.stats import entropy

PATH = "/home/stefan/stats/one_gram/"


def get_distr(path: str, col: str = "TRAIN_OCCURRENCES_COUNT") -> Dict[str, float]:
  df = pd.read_csv(path, sep="\t")
  symbols = df["SYMBOL"]
  df = df.set_index("SYMBOL")
  total_occ = np.sum(df[col][1:])
  distr_dict = OrderedDict({symb: df.loc[symb][col] / total_occ for symb in symbols[1:]})
  return distr_dict


def get_uniform_distr(keys_for_dict: List[str]) -> Dict[str, float]:
  n = len(keys_for_dict)
  uniform_dict = OrderedDict({key: 1 / n for key in keys_for_dict})
  return uniform_dict


def get_kld(distr_1, distr_2):
  assert distr_1.keys() == distr_2.keys()
  p_k = list(distr_1.values())
  q_k = list(distr_2.values())
  kld = entropy(p_k, q_k)
  return kld


def get_kld_for_all(names: List[str], keys):
  uniform_distr = get_uniform_distr(keys)
  kld_dict = {}
  for name in names:
    path = os.path.join(PATH, f"{name}.csv")
    if name == "stats_onegram_val":
      dist = get_distr(path, col="VAL_OCCURRENCES_COUNT")
    else:
      dist = get_distr(path)
    kld = get_kld(dist, uniform_distr)
    kld_dict[name] = kld
    print(name, ": ", kld)
  return kld_dict


full_df = pd.read_csv(os.path.join(PATH, "stats_onegram_full.csv"), sep="\t")
df_keys = full_df["SYMBOL"][1:]

names_list = ["stats_onegram_full", "stats_onegram_greedy_8h", "stats_onegram_kld_8h",
              "stats_onegram_rand1_8h", "stats_onegram_rand2_8h", "stats_onegram_rand3_8h", "stats_onegram_val"]

get_kld_for_all(names_list, df_keys)
# print(get_kld(get_distr(os.path.join(PATH, "stats_onegram_rand3_8h.csv")),
#      get_distr(os.path.join(PATH, "stats_onegram_kld_8h.csv"))))
