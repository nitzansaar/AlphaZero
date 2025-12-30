## this script goes through the training logs in ~/3x3_impl/output_3x3/logs and creates a graph of the loss

import os
import pandas as pd
import matplotlib.pyplot as plt

# get the training logs
logs = os.path.join(os.path.expanduser("~"), "3x3_impl", "output_3x3", "logs")

# get the loss files
loss_files = os.path.join(logs, "*.csv")

# read the loss files