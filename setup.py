from setuptools import setup, find_packages

setup(
    name = 'epimargin',
    version = '0.3',
    packages = ["epimargin"],
    author = "COVID International Working Group",
    description = "Toolkit for estimating epidemiological metrics and evaluating public health and economic policies.",
    url = "https://github.com/COVID-IWG/epimargin",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires = '>=3.6',
    install_requires = [
        "numpy",
        "pandas",
        "matplotlib",
        "arviz",
        "pymc3",
        "statsmodels",
        "flat-table",
        "requests",
        "seaborn",
        "tikzplotlib",
        "geopandas"
    ]
)
