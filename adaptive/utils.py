from pathlib import Path 
import sys

days  = 1
weeks = 7 

def cwd() -> Path:
    argv0 = sys.argv[0]
    if argv0.endswith("ipython"):
        return Path(".").resolve()
    return Path(argv0).resolve().parent
        