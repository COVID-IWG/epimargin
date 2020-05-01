from pathlib import Path 

days  = 1
weeks = 7 

def cwd() -> Path:
    try: 
        return Path(__file__).resolve().parent
    except NameError:
        return Path(".").resolve()