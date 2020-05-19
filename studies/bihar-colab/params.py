state_cases    = etl.load_cases("Bihar_Case_data_May18.csv")
district_cases = etl.split_cases_by_district(state_cases)
district_ts    = {district: etl.get_time_series(cases) for (district, cases) in district_cases.items()}
R_mandatory    = {district: estimate(district, ts, use_last = True) for (district, ts) in district_ts.items()}
districts, pops, migrations = etl.district_migration_matrix("Migration Matrix - District.csv")
for district in districts:
    if district not in R_mandatory.keys():
        R_mandatory[district] = 1.5

R_voluntary    = {district: 1.5*R for (district, R) in R_mandatory.items()}

migration_spike = etl.migratory_influx_matrix(data/"Bihar_state_district_migrants_matrix.xlsx - Table 1.csv", num_migrants, release_rate)

si, sf = 0, 1000

# first, 0% migration spike:
simulation_results = [ 
    run_policies(district_ts, pops, districts, migrations, None, gamma, R_mandatory, R_voluntary, seed = seed)
    for seed in tqdm(range(si, sf))
]

plot_simulation_range(simulation_results, ["31 May Release", "Adaptive Control"], etl.get_time_series(state_cases)["Hospitalized"])\
    .title("Bihar Policy Scenarios: Projected Infections (0% Migration Influx on 24 May)")\
    .xlabel("Date")\
    .ylabel("Number of Infections")\
    .annotate(f"data from May 18 | stochastic parameter range: ({si}, {sf}) | infectious period: {1/gamma} days | smoothing window: {window} | release rate: 0% | number of migrants {num_migrants} ")
plt.show()

# next, 50% migration spike:
simulation_results = [ 
    run_policies(district_ts, pops, districts, migrations, migration_spike, gamma, R_mandatory, R_voluntary, seed = seed)
    for seed in tqdm(range(si, sf))
]

plot_simulation_range(simulation_results, ["31 May Release", "Adaptive Control"], etl.get_time_series(state_cases)["Hospitalized"])\
    .title("Bihar Policy Scenarios: Projected Infections (50% Migration Influx on 24 May)")\
    .xlabel("Date")\
    .ylabel("Number of Infections")\
    .annotate(f"data from May 18 | stochastic parameter range: ({si}, {sf}) | infectious period: {1/gamma} days | smoothing window: {window} | release rate: {100*release_rate}% | number of migrants {num_migrants} ")\
    # .show()
plt.axvline(pd.to_datetime("24 May, 2020"), linestyle = "-.", color = "k", alpha = 0.2)
plt.show()

# projections
estimates = {district: estimate(district, ts, default = -1) for (district, ts) in district_ts.items()}
index = {k: v.last_valid_index() if v is not -1 else v for (k, v) in estimates.items()}
projections = []
for district, estimate in estimates.items():
    if estimate is -1:
        projections.append((district, None, None, None, None))
    else:
        idx = index[district]
        if idx is None or idx is -1:
            projections.append((district, None, None, None, None))
        else: 
            projections.append((district, *project(estimate.loc[idx])))
projdf = pd.DataFrame(data = projections, columns = ["district", "current R", "1 week projection", "2 week projection", "stderr"])
projdf.to_csv(figs/"bihar_may18_rt.csv")
