import pandas as pd
import numpy as np

file = 'waveform.data'

f = open(file)
# columns = f.readline().split()
# columns = ["label", 1, 2, 3, 4]
df = pd.DataFrame()
for line in f:
    df = df.append(pd.Series(data=line.split(",")), ignore_index=True)
# df.rename(columns={'16':'label'}, inplace=True)
df.to_csv(path_or_buf=file.split('.')[0]+'.csv')
