<h1 align="center">adaptive-control</h1>

<div align="center"> <img src="./docs/logo.svg" height="250"> </div>

<div align="center"> <img alt="Made With Python" src="https://img.shields.io/badge/made%20with-python-%233776AB.svg?style=for-the-badge&logo=python&logoColor=ffdf76"> <a href="https://twitter.com/miurbanchicago"><img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/miurbanchicago?logo=twitter&style=for-the-badge"></a> <img alt="MIT License" src="https://img.shields.io/github/license/mansueto-institute/adaptive-lockdown?logo=open-source-initiative&logoColor=white&style=for-the-badge"> </div>



# summary
an extension of a real-time Bayesian reproductive number estimation model for pandemic tracking that: 
1. runs a rolling regression to get a real time estimate of <i>R<sub>t</sub></i>
2. builds a graph of interactions between geographic units accounting for migratory (re-)introductions
3. at each <i>t</i> from <i>t<sub>i</sub></i> to <i>t<sub>f</sub></i>:

   - runs the standard SIRD forward epidemiological model 

   - simulates a migration out of each state to all other states (used as introductions at time <i>t</i>+1)
4. (optionally) projects SIRD curves under various policy scenarios and epidemiological parameters 

# organization
- `adaptive` - core package with model and estimator classes 
- `studies` - specific applications to geographies for which we have data

# data source notes
- India base demographic data provided by Anand Sahasranaman (Krea)
- India migration data from [Clement Imbert (Warwick)](https://warwick.ac.uk/fac/soc/economics/staff/cimbert/)
- India administrative data labels from [Sam Asher (JHU)](https://sais.jhu.edu/users/sasher2)
- Mumbai inter-ward travel from [Vaidehi Tandel (IDFC)](http://www.idfcinstitute.org/about/people/team/vaidehi-tandel/)
- Mumbai ward shape files from [Hindustan Times](https://github.com/HindustanTimesLabs/shapefiles/tree/master/city/mumbai/ward)
