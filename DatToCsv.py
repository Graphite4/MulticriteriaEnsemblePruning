import pandas as pd
import numpy as np

file = 'vowel0.dat'

f = open(file)
columns = f.readline().split(',')
columns = [c.strip() for c in columns]
# columns = ["label", 1, 2, 3, 4]
df = pd.DataFrame(columns=columns)
for line in f:
    df = df.append(pd.Series(data=line.strip().split(","), index=columns), ignore_index=True)
df.rename(columns={'Class':'label'}, inplace=True)
df.to_csv(path_or_buf=file.split('.')[0]+'.csv')
