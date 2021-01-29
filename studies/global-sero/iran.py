from adaptive.etl.csse import *
from adaptive.utils import setup
data, _ = setup()

csse_dir = data/"csse"
csse_dir.mkdir(exist_ok = True)

iran = data/"iran"
iran.mkdir(exist_ok = True)

load_country(csse_dir, "April 16, 2020", "July 3, 2020", "Iran")\
    .pipe(assemble_timeseries)\
    ["April 17, 2020":"July 2, 2020"]\
    .to_csv(iran/"Iran_1_natl.csv")