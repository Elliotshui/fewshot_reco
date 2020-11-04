import pandas as pd
import numpy as np
import json
from tqdm import tqdm

amazon_meta_data = []
amazon_meta_file = open("meta_Electronics.json")
lines = amazon_meta_file.readlines()
for l in tqdm(lines):
	amazon_meta_data.append(json.loads(l))

print(len(amazon_meta_data))

amazon_meta_df = pd.DataFrame.from_dict(amazon_meta_data)
