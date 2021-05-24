---
title: '`epimargin`: A Toolkit for Epidemiological Estimation, Simulation, and Policy Evaluation'
tags:
  - Python
  - epidemiology
  - stochastic processes
  - economics
  - COVID-19
  - Bayesian inference
authors:
  - name: Satej Soman 
    orcid: 0000-0001-8450-7025
    affiliation: 1
  - name: Caitlin Loftus
    affiliation: 1
  - name: Steven Buschbach
    affiliation: 1
  - name: Manasi Phadnis
    affiliation: 1
  - name: Nicholas Marchio
    orcid: 0000-0002-0677-1864
    affiliation: 1
  - name: Luis M. A. Bettencourt
    orcid: 0000-0001-6176-5160
    affiliation: "1, 2, 3, 4"
affiliations:
  - name: Mansueto Institute for Urban Innovation, University of Chicago
    index: 1
  - name: Department of Ecology & Evolution, University of Chicago
    index: 2
  - name: Department of Sociology, University of Chicago
    index: 3
  - name: Santa Fe Institute
    index: 4
date: 22 May 2021
bibliography: paper.bib

---

# Summary
As pandemics (including the COVID-19 crisis at time of writing) pose threats to societies, public health officials, epidemiologists, and policymakers need tools to assess the impact of disease, as well as a framework for understanding the effects and tradeoffs of health policy decisions. The `epimargin` package provides functionality to answer those questions in a way that incorporates irreducible uncertainty in both the input data and complex dynamics of disease propagation.  

The `epimargin` software package consists of: 

1. a set of Bayesian estimation procedures for epidemiological metrics such as the reproductive rate ($R_t$), which is the average number of secondary infections caused by an active infection

2. a flexible, stochastic epidemiological model informed by estimated metrics and reflecting real-world epidemic and geographic structure, and 

3. a set of tools to evaluate different public health policy choices simulated by the model.

The software is implemented in the Python 3 programming language and is built using commonly-used elements of the Python data science ecosystem, including NumPy [@harris2020array], Scipy [@virtanen2020scipy], and Pandas [@mckinney2011pandas].

# Statement of need

- data-driven analysis of policy choices

- data preparation choices made explicit and out of the box

- modularity and extensibility of methods and model 

- speed 

- visualization


The `epimargin` package has been used to drive a number of research projects and inform policy decisions in a number of countries:

1. lockdown, quarantine planning, migrant return policies, and vaccine distribution in India and Indonesia (at the behest of national governments, regional authorities, and various NGOs)

2. an illustration of a novel Bayesian estimator for the reproductive rate as well as general architectural principles for real-time epidemiological systems [@bettencourt2020systems]

3. a trigger-based algorithmic policy for determining when administrative units of a country should exit or return to a pandemic lockdown based on projected reproductive rates and case counts [@malani2020adaptive]

4. a World Bank study of vaccination policies in South Asia [@southasiavaccinates]

5. a general framework for quantifying the health and economic benefits to guide vaccine prioritization and distribution [@vaccineallocation]

<!-- `Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike. -->

# Comparison to other software packages
epiestim
epiforecasts
bubar
practical considerations
wallinga teunis
cori

# Figures
Sample output for common workflows are illustrated in the following figures:
## downloaded and cleaned time series
![Raw and cleaned case count timeseries for Mumbai downloaded from COVID19India.org.\label{fig:fig1}](fig_1.png)

## estimated reproductive rate
![Estimated reproductive rate over time for Mumbai](fig_2.png)

## forward projection/policy comparison
![Projected case counts using a stochastic compartmental model and reproductive rate estimates](fig_3.png)

<!--![Caption for example figure.](figure.png){ width=20% } -->

# Acknowledgements

We acknowledge code review and comments from Gyanendra Badgaiyan (IDFC Institute), ongoing conversations with Anup Malani (University of Chicago) and helpful discussions with Katelyn Gostic (University of Chicago) and Sarah Cobey (University of Chicago).

# References