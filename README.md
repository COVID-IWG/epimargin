<h1 align="center">adaptive-lockdown</h1>

<div align="center"> <img src="./docs/logo.svg" style="width:75%;"> </div>

[![python.org](https://img.shields.io/badge/made%20with-python-%233776AB.svg?style=for-the-badge&logo=python&logoColor=ffdf76)](https://www.python.org) &nbsp; [![Twitter Follow](https://img.shields.io/twitter/follow/miurbanchicago?logo=twitter&style=for-the-badge)](https://twitter.com/miurbanchicago) &nbsp; ![GitHub](https://img.shields.io/github/license/mansueto-institute/adaptive-lockdown?style=for-the-badge)


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
- India base demographic data provided by Anand Sahasranaman 
- India migration data from Clement Imbert
- India administrative data labels from Sam Asher
- Mumbai inter-ward travel from Vaidehi Tandel 
- Mumbai ward shape files from Hindustan Times
