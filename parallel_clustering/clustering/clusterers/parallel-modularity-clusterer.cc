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
#include <iomanip>
#include <iostream>

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
    modularity -= (resolution * degree * degree) / (total_edge_weight);
  }
  modularity = modularity / (total_edge_weight);
  return modularity;
}

absl::Status ParallelModularityClusterer::RefineClusters(
    const ClustererConfig& clusterer_config2,
    InMemoryClusterer::Clustering* initial_clustering) const {
  std::cout << "Begin modularity" << std::endl;

pbbs::timer t; t.start();
  // TODO: we just use correlation config
  const auto& config = clusterer_config2.correlation_clusterer_config();
  auto modularity_config= ComputeModularityConfig(graph_.Graph(), config.resolution());

  ClustererConfig clusterer_config;
  clusterer_config.CopyFrom(clusterer_config2);
  clusterer_config.mutable_correlation_clusterer_config()->set_resolution(std::get<1>(modularity_config));
  clusterer_config.mutable_correlation_clusterer_config()->set_edge_weight_offset(0);
t.stop(); t.reportTotal("Actual Modularity Config Time: ");
  auto status = ParallelCorrelationClusterer::RefineClusters(clusterer_config, initial_clustering,
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

double ParallelModularityClusterer::ComputeObjective2(
    const ClustererConfig& clusterer_config, 
  InMemoryClusterer::Clustering* initial_clustering) {
  auto n = graph_.Graph()->n;
  const auto& config = clusterer_config.correlation_clusterer_config();
  std::vector<double> shifted_edge_weight(n);

  std::vector<gbbs::uintE> cluster_ids(n);
  for (std::size_t i = 0; i < initial_clustering->size(); i++) {
    for (std::size_t j = 0; j < ((*initial_clustering)[i]).size(); j++) {
      cluster_ids[(*initial_clustering)[i][j]] = i;
    }
  }

  std::vector<gbbs::uintE> cluster_weights(n);
  for (std::size_t i = 0; i < n; i++) {
    cluster_weights[cluster_ids[i]]++;
  }

  // Compute cluster statistics contributions of each vertex
  pbbs::parallel_for(0, n, [&](std::size_t i) {
    gbbs::uintE cluster_id_i = cluster_ids[i];
    auto add_m = pbbslib::addm<double>();

    auto intra_cluster_sum_map_f = [&](gbbs::uintE u, gbbs::uintE v,
                                       float weight) -> double {
      // This assumes that the graph is undirected, and self-loops are counted
      // as half of the weight.
      if (cluster_id_i == cluster_ids[v])
        return (weight - config.edge_weight_offset()) / 2;
      return 0;
    };
    shifted_edge_weight[i] = graph_.Graph()->get_vertex(i).reduceOutNgh<double>(
        i, intra_cluster_sum_map_f, add_m);
  });
  double objective =
      parallel::ReduceAdd(absl::Span<const double>(shifted_edge_weight));

  auto resolution_seq = pbbs::delayed_seq<double>(n, [&](std::size_t i) {
    auto cluster_weight = cluster_weights[cluster_ids[i]];
    return 1 * (cluster_weight - 1);
  });
  objective -= config.resolution() * pbbslib::reduce_add(resolution_seq) / 2;

  return objective;
}

absl::StatusOr<InMemoryClusterer::Clustering>
ParallelModularityClusterer::Cluster(
    const ClustererConfig& clusterer_config) const {
  InMemoryClusterer::Clustering clustering(graph_.Graph()->n);

  // Create all-singletons initial clustering
  pbbs::parallel_for(0, graph_.Graph()->n, [&](std::size_t i) {
    clustering[i] = {static_cast<gbbs::uintE>(i)};
  });

  RETURN_IF_ERROR(RefineClusters(clusterer_config, &clustering));

  return clustering;
}

}  // namespace in_memory
}  // namespace research_graph
