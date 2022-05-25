# text-selection

[![PyPI](https://img.shields.io/pypi/v/text-selection.svg)](https://pypi.python.org/pypi/text-selection)
[![PyPI](https://img.shields.io/pypi/pyversions/text-selection.svg)](https://pypi.python.org/pypi/text-selection)
[![MIT](https://img.shields.io/github/license/stefantaubert/text-selection.svg)](https://github.com/stefantaubert/text-selection/blob/main/LICENSE)

CLI to select lines of a text file.

## Features

- dataset
  - create dataset from text
  - getting statistics
- subsets
  - add subsets
  - remove subsets
  - select entries by line number
  - select entries FIFO-style
  - select entries greedy-style
  - select entries greedy-style epoch-based
  - select entries kld-style
  - filter duplicates
  - filter lines via regex
  - filter lines via count of units
  - filter lines by weight
  - filter lines by unit vocabulary
  - filter lines by line text
  - filter lines by unit frequencies per line
  - sort entries by line number
  - sort entries by text
  - reverse entries
  - sort subsets after weights
  - export subsets lines
- subset
  - rename subset
- weights
  - create uniform weights
  - create weights from unit count
  - divide weights

## Roadmap

- add tests
- refactoring

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
- ordered-set >=4.1.0
- pronunciation-dictionary >=0.0.4

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
