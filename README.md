# text-selection

[![PyPI](https://img.shields.io/pypi/v/text-selection.svg)](https://pypi.python.org/pypi/text-selection)
[![PyPI](https://img.shields.io/pypi/pyversions/text-selection.svg)](https://pypi.python.org/pypi/text-selection)
[![MIT](https://img.shields.io/github/license/stefantaubert/text-selection.svg)](https://github.com/stefantaubert/text-selection/blob/main/LICENSE)

CLI to select lines of a text file.

## Features

- dataset
  - `create`: create a dataset based on a text file
  - `export-statistics`: exporting statistics to a CSV
- subsets
  - `add`: add subsets
  - `remove`: remove subsets
  - `rename`: rename subset
  - `select-all`: select all lines
  - `select-fifo`: select lines FIFO-style
  - `select-greedily`: select lines greedily regarding units
  - `select-greedily-ep`: select lines greedily regarding units (epoch-based)
  - `select-uniformly`: select lines with units uniformly distributed
  - `filter-duplicates`: filter duplicate lines
  - `filter-by-regex`: filter lines by regex
  - `filter-by-text`: filter lines by text
  - `filter-by-weight`: filter lines by weight
  - `filter-by-vocabulary`: filter lines by unit vocabulary
  - `filter-by-count`: filter lines by global unit frequencies
  - `filter-by-unit-freq`: filter lines by unit frequencies per line
  - `filter-by-line-nr`: filter lines by line number
  - `sort-by-line-nr`: sort lines by line number
  - `sort-by-text`: sort lines by text
  - `sort-by-weight`: sort lines by weights
  - `reverse`: reverse lines
  - `export`: export lines
- weights
  - `create-uniform`: create uniform weights
  - `create-from-count`: create weights from unit count
  - `divide`: divide weights

## Roadmap

- select/sort randomly
- add tests
- refactoring
- outsourcing greedy- and KLD-iterator

## Installation

```sh
pip install text-selection --user
```

## Usage

```sh
text-selection-cli
```

## Dependencies

- pandas
- tqdm
- scipy
- numpy
- ordered-set >=4.1.0

## License

MIT License

## Acknowledgments

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project-ID 416228727 – CRC 1410

## Citation

If you want to cite this repo, you can use this BibTeX-entry:

```bibtex
@misc{tsts22,
  author = {Taubert, Stefan},
  title = {text-selection},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/stefantaubert/text-selection}}
}
```
