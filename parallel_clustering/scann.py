import numpy as np
import h5py
import os
import requests
import tempfile
import time
import math

import scann
from sklearn import datasets as skdatasets

digits = skdatasets.load_digits()
dataset = digits.data
target_id = digits.target
dataset_shape = dataset.shape
print(dataset_shape)

# create scann
normalized_dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]
nl = int(math.sqrt(dataset_shape[0]))
searcher = scann.scann_ops.build(normalized_dataset, 10, "dot_product").tree(
  num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(
  2, anisotropic_quantization_threshold=0.2).reorder(100).build()
neighbors, distances = searcher.search_batched(dataset, final_num_neighbors=51)

# output neighbor list
with open("/home/jeshi/snap/digits_50nn", "a+") as out_file:
  for index, row in enumerate(neighbors):
    for index_x, x in enumerate(row):
      if x != index:
        out_file.write(str(index) + " " + str(x) + " " + str(distances[index][index_x]) + "\n")

# output target
with open("/home/jeshi/snap/digits_targetid", "a+") as outfile:
  for x in target_id:
    print(str(x) + "\n")

