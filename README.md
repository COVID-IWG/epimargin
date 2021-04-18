<h1 align="center">epimargin</h1>

<div align="center"> <img src="./docs/logo.svg" height="250"> </div>

<!-- <div align="center"> <img alt="Made With Python" src="https://img.shields.io/badge/made%20with-python-%233776AB.svg?style=for-the-badge&logo=python&logoColor=ffdf76"> </div> -->

a public health policy analysis toolkit consisting of: 
1. Bayesian simulated annealing estimator for the reproductive rate (<i>R<sub>t</sub></i>)
2. a stochastic compartmental model class supporting multiple compartments schemes (SIR, SEIR, SIRV, etc.)
3. policy impact evaluation that calculates longevity benefits and economic activity disruption/resumption under scenarios including: 
    - targeted lockdowns
    - urban-rural migration and reverse-migration flows
    - multiple vaccine allocation prioritizations 

# organization
- `adaptive` - core package with model and estimator classes 
- `studies` - specific applications to geographies and policy evaluation exercises

# data source notes
- India base demographic data provided by Anand Sahasranaman (Krea)
- India migration data from [Clement Imbert (Warwick)](https://warwick.ac.uk/fac/soc/economics/staff/cimbert/)
- India administrative data labels from [Sam Asher (JHU)](https://sais.jhu.edu/users/sasher2)
- Mumbai inter-ward travel from [Vaidehi Tandel (IDFC)](http://www.idfcinstitute.org/about/people/team/vaidehi-tandel/)
- Mumbai ward shape files from [Hindustan Times](https://github.com/HindustanTimesLabs/shapefiles/tree/master/city/mumbai/ward)
