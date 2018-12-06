import tensorflow as tf
import numpy as np
import time
import datetime
import os
from model import CNN
import data_helper
from tensorflow.contrib import learn

test_sample_percentage = 0.2
data_path = 'normalized.txt'

