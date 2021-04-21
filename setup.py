from setuptools import find_packages, setup

setup(
    dependency_links=[
    ],
    name="text_selection",
    version="1.0.0",
    url="https://github.com/stefantaubert/text-selection.git",
    author="Stefan Taubert",
    author_email="stefan.taubert@posteo.de",
    description="Utils for text selection",
    packages=["text_selection"],
    install_requires=[
        "ordered-set",
        "tqdm",
        "sklearn",
        "numpy",
    ],
)
