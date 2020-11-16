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

#include "clustering/clusterers/parallel-modularity-clusterer.h"

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "clustering/clusterers/parallel-correlation-clusterer.h"
#include "clustering/config.pb.h"
#include "clustering/gbbs-graph.h"
#include "clustering/in-memory-clusterer.h"
#include "parallel/parallel-graph-utils.h"
#include "clustering/status_macros.h"

namespace research_graph {
namespace in_memory {

double ComputeModularity(
InMemoryClusterer::Clustering& initial_clustering,
gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
double total_edge_weight, std::vector<gbbs::uintE>& cluster_ids,
double resolution){
  total_edge_weight = 0;
  double modularity = 0;
  for (std::size_t i = 0; i < graph.n; i++) {
    auto vtx = graph.get_vertex(i);
    auto nbhrs = vtx.getOutNeighbors();
    double deg_i = vtx.getOutDegree();
    for (std::size_t j = 0; j < deg_i; j++) {
      total_edge_weight++;
      auto nbhr = std::get<0>(nbhrs[j]);
      //double deg_nbhr = graph.get_vertex(nbhr).getOutDegree();
      if (cluster_ids[i] == cluster_ids[nbhr]) {
        modularity++;
      }
    }
  }
  //modularity = modularity / 2; // avoid double counting
  for (std::size_t i = 0; i < initial_clustering.size(); i++) {
    double degree = 0;
    for (std::size_t j = 0; j < initial_clustering[i].size(); j++) {
      auto vtx_id = initial_clustering[i][j];
      auto vtx = graph.get_vertex(vtx_id);
      degree += vtx.getOutDegree();
    }
    modularity -= (resolution * degree * degree) / total_edge_weight;
  }
  modularity = modularity / total_edge_weight;
  return modularity;
}

absl::Status ParallelModularityClusterer::RefineClusters(
    const ClustererConfig& clusterer_config2,
    InMemoryClusterer::Clustering* initial_clustering) const {
  // TODO: we just use correlation config
  const auto& config = clusterer_config2.correlation_clusterer_config();

  ClustererConfig clusterer_config;
  std::size_t total_edge_weight = 0;
  std::vector<double> node_weights(graph_.Graph()->n);
  for (std::size_t i = 0; i < graph_.Graph()->n; i++) {
    auto vtx = graph_.Graph()->get_vertex(i);
    auto wgh = vtx.getOutDegree();
    // TODO: this assumes unit edge weights
    total_edge_weight += wgh;
    node_weights[i] = wgh;
  }
  double resolution = config.resolution() / (total_edge_weight);
  clusterer_config.mutable_correlation_clusterer_config()->set_resolution(resolution);
  clusterer_config.mutable_correlation_clusterer_config()->set_edge_weight_offset(0);
  clusterer_config.mutable_correlation_clusterer_config()->set_clustering_moves_method(
    clusterer_config2.correlation_clusterer_config().clustering_moves_method()
  );
  clusterer_config.mutable_correlation_clusterer_config()->set_subclustering_method(
    clusterer_config2.correlation_clusterer_config().subclustering_method()
  );

  //ParallelCorrelationClusterer correlation_clusterer;
  //auto graph = correlation_clusterer.MutableGraph();
  //graph = &graph_;

  //auto status = correlation_clusterer.RefineClusters(clusterer_config, initial_clustering,
  //  std::move(node_weights));

  auto status = ParallelCorrelationClusterer::RefineClusters(clusterer_config, initial_clustering,
    node_weights);

  std::vector<gbbs::uintE> cluster_ids(graph_.Graph()->n);
  for (std::size_t i = 0; i < initial_clustering->size(); i++) {
    for (std::size_t j = 0; j < ((*initial_clustering)[i]).size(); j++) {
      cluster_ids[(*initial_clustering)[i][j]] = i;
    }
  }

  double modularity = ComputeModularity(*initial_clustering,
    *graph_.Graph(), total_edge_weight, cluster_ids, config.resolution());
  std::cout << "Modularity: " << modularity << std::endl;

  return absl::OkStatus();
}

double ParallelModularityClusterer::ComputeModularity2(const ClustererConfig& clusterer_config, 
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
ParallelModularityClusterer::Cluster(
    const ClustererConfig& clusterer_config) const {
  InMemoryClusterer::Clustering clustering(graph_.Graph()->n);

  // Create all-singletons initial clustering
  pbbs::parallel_for(0, graph_.Graph()->n, [&](std::size_t i) {
    clustering[i] = {static_cast<int32_t>(i)};
  });

  RETURN_IF_ERROR(RefineClusters(clusterer_config, &clustering));

  return clustering;
}

}  // namespace in_memory
}  // namespace research_graph