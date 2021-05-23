<h1 align="center">epimargin</h1>

<div align="center"> <img src="./docs/logo.svg" height="150"> </div>

<hr>

a public health policy analysis toolkit consisting of: 
1. Bayesian simulated annealing estimator for the reproductive rate (<i>R<sub>t</sub></i>)
2. a stochastic compartmental model class supporting multiple compartment schemes (SIR, SEIR, SIRV, subpopulations, etc.)
3. policy impact evaluation that calculates longevity benefits and economic activity disruption/resumption under scenarios including: 
    - targeted lockdowns
    - urban-rural migration and reverse-migration flows
    - multiple vaccine allocation prioritizations 

# organization
- `epimargin` - core package with model and estimator classes 
- `studies` - specific applications to geographies and policy evaluation exercises

# installation  

## end user
The `epimargin` package is available on PyPI and can be installed via `pip`: 
    
    pip install epimargin

## development 
To contribute, clone the repository, install in editable mode, and install the development dependencies: 

    git clone https://github.com/COVID-IWG/covid-metrics-infra 
    cd covid-metrics-infra 

    pip install -r requirements.txt
    pip install -e . 

We recommend using a virtual environment for development.

# tutorial 
In this tutorial, we will download data from COVID19India.org, estimate the reproductive rate over time, plug these estimates into a compartmental model, and compare two policy scenarios.

## download and clean data 
## estimate <i>R<sub>t</sub></i>
## set up a model and run it forward
## compare policy scenarios