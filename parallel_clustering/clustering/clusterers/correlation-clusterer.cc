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

#include "clustering/clusterers/correlation-clusterer.h"

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>
#include <iomanip>
#include <iostream>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "clustering/clusterers/parallel-correlation-clusterer-internal.h"
#include "clustering/config.pb.h"
#include "clustering/gbbs-graph.h"
#include "clustering/in-memory-clusterer.h"
#include "parallel/parallel-graph-utils.h"
#include "clustering/status_macros.h"

#include "external/gbbs/benchmarks/KCore/JulienneDBS17/KCore.h"
#include "external/gbbs/gbbs/pbbslib/sparse_additive_map.h"
#include "external/gbbs/pbbslib/random_shuffle.h"
//#include "external/gbbs/pbbslib/union_find.h"

#include "clustering/clusterers/correlation-clusterer-internal.h"

namespace research_graph {
namespace in_memory {


absl::Status CorrelationClusterer::RefineClusters(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering) const {
  std::vector<double> empty;
  return RefineClusters(clusterer_config, initial_clustering, empty);
}

absl::Status CorrelationClusterer::RefineClusters(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    gbbs::symmetric_graph<gbbs::csv_bytepd_amortized, pbbslib::empty>* graph) const {
  std::vector<double> empty;
  return RefineClusters(clusterer_config, initial_clustering, empty, graph);
}

absl::Status CorrelationClusterer::RefineClusters(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    std::vector<double> node_weights, double resolution) const {
    return RefineClusters(clusterer_config, initial_clustering, node_weights, graph_.Graph(), resolution);
}

absl::Status CorrelationClusterer::RefineClusters_subroutine(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering) const {
  std::vector<double> empty;
  return RefineClusters_subroutine(clusterer_config, initial_clustering, empty);
}


absl::Status CorrelationClusterer::RefineClusters_subroutine(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    std::vector<double> node_weights) const {
    return RefineClusters_subroutine(clusterer_config, initial_clustering, node_weights, graph_.Graph());
}



// Note: only allows for refinement + clustering; no subclustering, no kcore
absl::Status CorrelationClusterer::RefineClusters(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    std::vector<double> node_weights,
    gbbs::symmetric_graph<gbbs::csv_bytepd_amortized, pbbslib::empty>* graph, double original_resolution) const {
  const auto& config = clusterer_config.correlation_clusterer_config();
  RETURN_IF_ERROR(RefineClusters_subroutine(clusterer_config, initial_clustering, node_weights, graph));
  return absl::OkStatus();
}

absl::StatusOr<InMemoryClusterer::Clustering> CorrelationClusterer::Cluster(
    const ClustererConfig& clusterer_config) const {
  InMemoryClusterer::Clustering clustering(graph_.Graph()->n);

  // Create all-singletons initial clustering
  for (std::size_t i = 0; i < graph_.Graph()->n; i++) {
    clustering[i] = {static_cast<gbbs::uintE>(i)};
  }

  RETURN_IF_ERROR(RefineClusters(clusterer_config, &clustering));

  return clustering;
}

}  // namespace in_memory
}  // namespace research_graph
