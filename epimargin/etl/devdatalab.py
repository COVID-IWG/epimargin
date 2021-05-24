from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

""" set up migration matrices from DDL matrix data """

district_2011_replacements = {
    'Maharashtra' : {
        'Mumbai Suburban' : 'Mumbai'}
 }

def load_migration_matrix(matrix_path: Path, populations: np.array) -> np.matrix:
    M  = np.loadtxt(matrix_path, delimiter=',') # read in raw data
    M *= populations[:,  None]                  # weight by population
    M /= M.sum(axis = 0)                        # normalize
    return M

def district_migration_matrices(
    matrix_path: Path, 
    states: Sequence[str]) -> Dict[str, np.matrix]:
    mm = pd.read_csv(matrix_path)
    aggregations = dict()
    for col in  ['D_StateCensus2011', 'D_DistrictCensus2011', 'O_StateCensus2011', 'O_DistrictCensus2011']:
        mm[col] = mm[col].str.title().str.replace("&", "and")
    for state in  states:
        mm_state = mm[(mm.D_StateCensus2011 == state) & (mm.O_StateCensus2011 == state)]
        # handle states that need migration data combined (e.g. Mumbai and Mumbai Suburban)
        if state in district_2011_replacements:
            mm_state.replace(district_2011_replacements[state], inplace=True)
        # group to combine multiple districts with same name based on above
        grouped_mm_state = mm_state.groupby(['D_DistrictCensus2011', 'O_DistrictCensus2011'])[['O_Population_2011','NSS_STMigrants']].sum().reset_index()
        pivot = grouped_mm_state.pivot(index = "D_DistrictCensus2011", columns = "O_DistrictCensus2011", values = "NSS_STMigrants").fillna(0)
        M  = np.matrix(pivot)
        Mn = M/M.sum(axis = 0)
        Mn[np.isnan(Mn)] = 0
        aggregations[state] = (
            pivot.index, 
            grouped_mm_state.groupby("O_DistrictCensus2011")["O_Population_2011"].agg(lambda x: list(x)[0]).values, 
            Mn
        )
    return aggregations