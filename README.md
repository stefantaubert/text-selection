# text-selection

[![PyPI](https://img.shields.io/pypi/v/text-selection.svg)](https://pypi.python.org/pypi/text-selection)
[![PyPI](https://img.shields.io/pypi/pyversions/text-selection.svg)](https://pypi.python.org/pypi/text-selection)
[![MIT](https://img.shields.io/github/license/stefantaubert/text-selection.svg)](https://github.com/stefantaubert/text-selection/blob/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/wheel/text-selection.svg)](https://pypi.python.org/pypi/text-selection/#files)
![PyPI](https://img.shields.io/pypi/implementation/text-selection.svg)
[![PyPI](https://img.shields.io/github/commits-since/stefantaubert/text-selection/latest/master.svg)](https://github.com/stefantaubert/text-selection/compare/v0.0.3...master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7984739.svg)](https://doi.org/10.5281/zenodo.7984739)

Command-line interface (CLI) to select lines of a text file.

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
  - `select-randomly`: select lines randomly
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
  - `sort-by-shuffle`: shuffle lines
  - `reverse`: reverse lines
  - `export`: export lines
- weights
  - `create-from-file`: create weights from file
  - `create-uniform`: create uniform weights
  - `create-from-count`: create weights from unit count
  - `divide`: divide weights

## Roadmap

- add tests
- refactoring
- outsourcing greedy- and KLD-iterator

## Installation

```sh
pip install text-selection --user
```

## Usage

```txt
usage: text-selection-cli [-h] [-v] {dataset,subsets,weights} ...

CLI to select lines of a text file.

positional arguments:
  {dataset,subsets,weights}  description
    dataset                  dataset commands
    subsets                  subsets commands
    weights                  weights commands

optional arguments:
  -h, --help                 show this help message and exit
  -v, --version              show program's version number and exit
```

## Dependencies

- `tqdm`
- `numpy`
- `scipy`
- `pandas`
- `ordered_set>=4.1.0`

## Contributing

If you notice an error, please don't hesitate to open an issue.

### Development setup

```sh
# update
sudo apt update
# install Python 3.8, 3.9, 3.10 & 3.11 for ensuring that tests can be run
sudo apt install python3-pip \
  python3.8 python3.8-dev python3.8-distutils python3.8-venv \
  python3.9 python3.9-dev python3.9-distutils python3.9-venv \
  python3.10 python3.10-dev python3.10-distutils python3.10-venv \
  python3.11 python3.11-dev python3.11-distutils python3.11-venv
# install pipenv for creation of virtual environments
python3.8 -m pip install pipenv --user

# check out repo
git clone https://github.com/stefantaubert/text-selection.git
cd text-selection
# create virtual environment
python3.8 -m pipenv install --dev
```

## Running the tests

```sh
# first install the tool like in "Development setup"
# then, navigate into the directory of the repo (if not already done)
cd text-selection
# activate environment
python3.8 -m pipenv shell
# run tests
tox
```

Final lines of test result output:

```log
  py38: commands succeeded
  py39: commands succeeded
  py310: commands succeeded
  py311: commands succeeded
  congratulations :)
```

## License

MIT License

## Acknowledgments

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project-ID 416228727 – CRC 1410

## Citation

If you want to cite this repo, you can use this BibTeX-entry generated by GitHub (see *About => Cite this repository*).

## Changelog

- v0.0.3 (2023-05-30)
  - Changed
    - Improved speed for filtering OOV/IV words by up to ~20k words/s
  - Added
    - Added `subsets select-randomly`
    - Added `subsets sort-by-shuffle`
    - Added `subsets add` option `--skip-existing`
  - Bugfix
    - Fixed evaluation of "from subsets" to ensure that the subsets exist
    - Fixed `subsets remove` didn't worked
- v0.0.2 (2023-01-13)
  - Added
    - Added creation of weights from lines
    - Add `--limit` to select duplicates
    - Add exit code
  - Changed
    - Set `--limit` positional where applicable
    - Don't output expected warning from `numpy` on KLD selection
  - Bugfixes
- v0.0.1 (2022-05-25)
  - Initial release
