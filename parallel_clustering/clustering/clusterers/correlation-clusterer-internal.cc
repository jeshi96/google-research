// Copyright 2020 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "clustering/clusterers/correlation-clusterer-internal.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "clustering/clusterers/correlation-clusterer-util.h"
#include "clustering/config.pb.h"
#include "external/gbbs/gbbs/bridge.h"
#include "external/gbbs/gbbs/gbbs.h"
#include "external/gbbs/gbbs/macros.h"
#include "external/gbbs/pbbslib/random_shuffle.h"
#include "external/gbbs/pbbslib/sample_sort.h"
#include "external/gbbs/pbbslib/seq.h"
#include "external/gbbs/pbbslib/sequence_ops.h"
#include "external/gbbs/pbbslib/utilities.h"
#include "clustering/in-memory-clusterer.h"
#include "parallel/parallel-graph-utils.h"
#include "parallel/parallel-sequence-ops.h"
#include "external/gbbs/gbbs/pbbslib/sparse_table.h"

#include "external/gbbs/benchmarks/Connectivity/WorkEfficientSDB14/Connectivity.h"
#include "clustering/clusterers/parallel-correlation-clusterer-internal.h"

namespace research_graph {
namespace in_memory {

using NodeId = gbbs::uintE;
using ClusterId = SeqClusteringHelper::ClusterId;

void SeqClusteringHelper::ResetClustering(
  const std::vector<std::vector<gbbs::uintE>>& clustering) {
  for (std::size_t i = 0; i < num_nodes_; i++) {
      cluster_weights_[i] = 0;
      cluster_sizes_[i] = 0;
      
  }
  SetClustering(clustering);
}

void SeqClusteringHelper::SetClustering(
    const std::vector<std::vector<gbbs::uintE>>& clustering) {
  if (clustering.empty()) {
    for (std::size_t i = 0; i < num_nodes_; i++) {
      cluster_sizes_[i] = 1;
      cluster_ids_[i] = i;
      cluster_weights_[i] = node_weights_[i];
    }
  } else {
    for (std::size_t i = 0; i < clustering.size(); i++) {
      cluster_sizes_[i] = clustering[i].size();
      for (auto j : clustering[i]) {
        cluster_ids_[j] = i;
        cluster_weights_[i] += node_weights_[j];
      }
    }
  }
}

double SeqClusteringHelper::NodeWeight(NodeId id) const {
  return id < node_weights_.size() ? node_weights_[id] : 1.0;
}




}  // namespace in_memory
}  // namespace research_graph