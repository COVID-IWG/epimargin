import pandas as pd
from adaptive.utils import setup, mkdir

data, _ = setup()
peru = mkdir(data/"peru")

schema = dict(zip(
    "FECHA_CORTE;UUID;FECHA_FALLECIMIENTO;EDAD_DECLARADA;SEXO;FECHA_NAC;DEPARTAMENTO;PROVINCIA;DISTRITO".split(";"), 
    "CUT_DATE;UUID;DEATH_DATE;DECLARED_AGE;SEX;BIRTH_DATE;DEPARTMENT;PROVINCE;DISTRICT".lower().split(";")
))  

#source: https://www.datosabiertos.gob.pe/dataset/fallecidos-por-covid-19-ministerio-de-salud-minsa
df = pd.read_csv(peru/"fallecidos_covid.csv", delimiter = ";", encoding = "ISO-8859-1")\
        .rename(columns = schema)

df[df.district == "IQUITOS"][["death_date", "declared_age", "sex", "district"]]\
    .assign(
        death_date = lambda _: pd.to_datetime(_["death_date"].astype(str).str.lstrip("0"), format = "%Y%m%d"),
        sex        = lambda _: _["sex"].str[0]
    )\
    .to_csv(peru/"iquitos.csv")