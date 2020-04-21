# adaptive-bayesian-controls

# summary
an extension of the real-time Bayesian reproductive number model that: 
1. runs a rolling regression 
2. builds a graph of interactions between geographic units accounting for migratory (re-)introductions
3. at each <i>t</i> from <i>t<sub>i</sub></i> to <i>t<sub>f</sub></i>:
   a. simulates a migration out of each state to all other states
   b. runs forward an epidemiological model 

## todo

### bayesian PID controller

### LOESS smoothing

### Gaussian process regression for small-<i>n</i> units 

### likelihood update options:
1. classic, all-history (Bettencourt and Ribeiro 2008)
2. rolling window ([recent Bettencourt work](https://github.com/mansueto-institute/Rt_Real-time_Estimation_Case_Prediction), as well as Systrom 2020 : [iPython notebook](https://github.com/k-sys/covid-19/blob/master/Realtime%20R0.ipynb), [Medium post](http://systrom.com/blog/the-metric-we-need-to-manage-covid-19/), [real-time <i>R</i> estimates for US states](https://rt.live/))
3. exponential decay (novel here)



## data source notes
- India demographic data provided by Anand Sahasranaman 