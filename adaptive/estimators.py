from collections import SimpleNamespace as namespace
from typing import Sequence


schemas = namespace(default = None, india_v1 = None, india_v2 = None)
methods = namespace()

class Schema: 
    def __init__(self, columns: Sequence[str], S_key: str, I_key: str, R_key: str, D_key: str):
        self.columns = columns
        self.S_key   = S_key
        self.I_key   = I_key
        self.R_key   = R_key
        self.D_key   = D_key
    
    