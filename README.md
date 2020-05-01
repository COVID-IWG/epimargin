<h1 align="center">adaptive-lockdown</h1>

# summary
an extension of the real-time Bayesian reproductive number model that: 
1. runs a rolling regression to get a real time estimate of <i>R(t)</i>
2. builds a graph of interactions between geographic units accounting for migratory (re-)introductions
3. at each <i>t</i> from <i>t<sub>i</sub></i> to <i>t<sub>f</sub></i>:

   - runs the standard SIRD forward epidemiological model 

   - simulates a migration out of each state to all other states (used as introductions at time <i>t+1</i>)

# organization
- `adaptive` - core package with model and estimator classes 
- `studies` - specific applications to geographies for which we have data

# data source notes
- India base demographic data provided by Anand Sahasranaman 
- 