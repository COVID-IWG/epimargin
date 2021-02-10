# Vaccine Allocation Priorities Using Surveillance and Economic Data

code and analysis for our model-informed vaccine distribution proposal

## files used in paper

- `common_TN_data.py`: downloads and sets up data for Tamil Nadu
- `hazard_plot.py`: plots incremental probability of death in Tamil Nadu 
- `national_model.py`: runs national model as superposition of 730 district models, or one population bin
- `vaccination_policies.py`: state-level vaccine policies 
- `vaccine_mortality_district.py`: calculates explicit hazards for various vaccination plans
- `vaccine_policies_district.py`: runs vaccination plans for each district in Tamil Nadu

## prior experiments
- `scaled_hazards.py`: hazards run by seroprevalence scaling
- `agestruct.py`: explicitly age-structured model (not ex post facto age bin assignment)
- `test_pos_scaling.py`: attempts to scale confirmed cases by accounting for test positivity
- `OWID_test_scaling.py`: attempts to scale confirmed cases by accounting for test positivity on OWID data 