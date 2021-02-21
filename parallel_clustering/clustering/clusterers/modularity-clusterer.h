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

#ifndef PARALLEL_CLUSTERING_CLUSTERERS_MODULARITY_CLUSTERER_H_
#define PARALLEL_CLUSTERING_CLUSTERERS_MODULARITY_CLUSTERER_H_

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

using Clustering = std::vector<std::vector<gbbs::uintE>>;

// A local-search based clusterer optimizing the correlation clustering
// objective. See comment above CorrelationClustererConfig in
// ../config.proto for more. This uses the CorrelationClustererConfig proto.
// Also, note that the input graph is required to be undirected.
template<class ClusterGraph>
class ModularityClusterer : public CorrelationClusterer<ClusterGraph> {
 public:
  GbbsGraph<ClusterGraph> graph_;
  using ClusterId = gbbs::uintE;


  // initial_clustering must include every node in the range
  // [0, MutableGraph().NumNodes()) exactly once.
  absl::Status RefineClusters(const ClustererConfig& clusterer_config2,
                              Clustering* initial_clustering) const override {
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
  auto status = CorrelationClusterer<ClusterGraph>::RefineClusters(clusterer_config, initial_clustering,
    std::get<0>(modularity_config), config.resolution());

  std::vector<gbbs::uintE> cluster_ids(graph_.Graph()->n);
  for (std::size_t i = 0; i < initial_clustering->size(); i++) {
    for (std::size_t j = 0; j < ((*initial_clustering)[i]).size(); j++) {
      cluster_ids[(*initial_clustering)[i][j]] = i;
    }
  }

  double modularity = ComputeModularity(*initial_clustering,
    *graph_.Graph(), std::get<2>(modularity_config), cluster_ids, config.resolution());
  std::cout << "Modularity: " << std::setprecision(17) << modularity << std::endl;

  return absl::OkStatus();
}

  absl::StatusOr<Clustering> Cluster(
      const ClustererConfig& config) const override{
  Clustering clustering(graph_.Graph()->n);

  // Create all-singletons initial clustering
  for (std::size_t i = 0; i < graph_.Graph()->n; i++) {
    clustering[i] = {static_cast<gbbs::uintE>(i)};
  }

  RETURN_IF_ERROR(RefineClusters(clusterer_config, &clustering));

  return clustering;
}

  
};

}  // namespace in_memory
}  // namespace research_graph

#endif  // PARALLEL_CLUSTERING_CLUSTERERS_MODULARITY_CLUSTERER_H_
