# adaptive-bayesian-controls

# summary
a two-stage extension of the SIR model: 
1. a PID-style control for modulating the growth rate <i>R</i> within a geographic unit
2. a graph of interactions between geographic units accounting for migratory (re-)introductions

## todo - likelihood update options:
1. classic (Bettencourt and Ribeiro 2008)
2. rolling window (Systrom 2020 | [iPython notebook](https://github.com/k-sys/covid-19/blob/master/Realtime%20R0.ipynb) [Medium post](http://systrom.com/blog/the-metric-we-need-to-manage-covid-19/) [real-time <i>R</i> estimates for US states](https://rt.live/))
3. exponential decay (novel here)

## data source notes
- India demographic data provided by Anand Sahasranaman 