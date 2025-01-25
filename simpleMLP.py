
import torch #Pytorch
import matplotlib as plt #Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns #Seaborn
import numpy as np
import pandas as pd
import math as math
import os
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split # Splits data into training/testing sets
from sklearn.preprocessing import StandardScaler #scales data for better preformance

df = pd.read_csv('EditedCSVs/df2CSV.csv')
