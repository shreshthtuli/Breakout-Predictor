import pandas as pd
import numpy as np
from sys import argv
import random

a = pd.read_csv(argv[1], header=None).values
b = pd.read_csv(argv[2], header=None).values

# print(a.shape, b.shape)
print(str(a[0,0]+","+str(a[0,1])))

for i in range(1, a.shape[0]):
    choice = random.choice([a[i,1], b[i,1]])
    print(str(i-1)+","+str(choice))