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

#include "clustering/clusterers/modularity-clusterer.h"

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "clustering/clusterers/correlation-clusterer.h"
#include "clustering/config.pb.h"
#include "clustering/gbbs-graph.h"
#include "clustering/in-memory-clusterer.h"
#include "parallel/parallel-graph-utils.h"
#include "clustering/status_macros.h"

#include "clustering/clusterers/parallel-modularity-clusterer.h"

namespace research_graph {
namespace in_memory {

absl::Status ModularityClusterer::RefineClusters(
    const ClustererConfig& clusterer_config2,
    InMemoryClusterer::Clustering* initial_clustering) const {
  std::cout << "Begin modularity" << std::endl;
pbbs::timer t; t.start();
  // TODO: we just use correlation config
  const auto& config = clusterer_config2.correlation_clusterer_config();
  auto modularity_config= SeqComputeModularityConfig(graph_.Graph(), config.resolution());

  ClustererConfig clusterer_config;
  clusterer_config.CopyFrom(clusterer_config2);
  clusterer_config.mutable_correlation_clusterer_config()->set_resolution(std::get<1>(modularity_config));
  clusterer_config.mutable_correlation_clusterer_config()->set_edge_weight_offset(0);
t.stop(); t.reportTotal("Actual Modularity Config Time: ");
  auto status = CorrelationClusterer::RefineClusters(clusterer_config, initial_clustering,
    std::get<0>(modularity_config), config.resolution());

  std::vector<gbbs::uintE> cluster_ids(graph_.Graph()->n);
  for (std::size_t i = 0; i < initial_clustering->size(); i++) {
    for (std::size_t j = 0; j < ((*initial_clustering)[i]).size(); j++) {
      cluster_ids[(*initial_clustering)[i][j]] = i;
    }
  }

  double modularity = ComputeModularity(*initial_clustering,
    *graph_.Graph(), std::get<2>(modularity_config), cluster_ids, config.resolution());
  std::cout << "Modularity: " << modularity << std::endl;

  return absl::OkStatus();
}

double ModularityClusterer::ComputeModularity2(const ClustererConfig& clusterer_config, 
  InMemoryClusterer::Clustering* initial_clustering) {
    const auto& config = clusterer_config.correlation_clusterer_config();
  std::size_t total_edge_weight = 0;
  for (std::size_t i = 0; i < graph_.Graph()->n; i++) {
    auto vtx = graph_.Graph()->get_vertex(i);
    auto wgh = vtx.getOutDegree();
    // TODO: this assumes unit edge weights
    total_edge_weight += wgh;
  }
  std::vector<gbbs::uintE> cluster_ids(graph_.Graph()->n);
  for (std::size_t i = 0; i < initial_clustering->size(); i++) {
    for (std::size_t j = 0; j < ((*initial_clustering)[i]).size(); j++) {
      cluster_ids[(*initial_clustering)[i][j]] = i;
    }
  }
  double modularity = ComputeModularity(*initial_clustering,
    *graph_.Graph(), total_edge_weight, cluster_ids, config.resolution());
  return modularity;
}

absl::StatusOr<InMemoryClusterer::Clustering>
ModularityClusterer::Cluster(
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
