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

#ifndef RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_CORRELATION_CLUSTERER_INTERNAL_H_
#define RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_CORRELATION_CLUSTERER_INTERNAL_H_

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "clustering/config.pb.h"
#include "external/gbbs/gbbs/graph.h"
#include "external/gbbs/gbbs/vertex_subset.h"
#include "clustering/in-memory-clusterer.h"

#include "parallel-correlation-clusterer-internal.h"
#include "clustering/clusterers/parallel-correlation-clusterer-internal.h"

namespace research_graph {
namespace in_memory {


float FloatFromWeightCCI(float weight);
float FloatFromWeightCCI(pbbslib::empty weight);

// Retrieves a list of inter-cluster edges, given a set of cluster_ids
// that form the vertices of a new graph. Maps all edges in original_graph
// to cluster ids. Depending on is_valid_func, combines edges in the same
// cluster to be a self-loop.
template<class G>
std::vector<absl::flat_hash_map<gbbs::uintE, float>>
SeqRetrieveInterClusterEdges(
    G& original_graph,
    const std::vector<gbbs::uintE>& cluster_ids,
    const std::function<bool(gbbs::uintE, gbbs::uintE)>& is_valid_func, gbbs::uintE max_cluster_id,
    const std::function<float(float, float)>& aggregate_func) {
  using W = typename G::weight_type;
  std::vector<absl::flat_hash_map<gbbs::uintE, float>> all_edges(max_cluster_id + 1);

  for (std::size_t j = 0; j < original_graph.n; j++) {
    auto vtx = original_graph.get_vertex(j);
    if (cluster_ids[j] != UINT_E_MAX) {
      auto map_f = [&](gbbs::uintE u, gbbs::uintE v, W weight) {
        if (is_valid_func(cluster_ids[v], cluster_ids[u]) &&
            cluster_ids[v] != UINT_E_MAX 
            && (v <= u || cluster_ids[v] != cluster_ids[u])
            ) {
          auto element = all_edges[cluster_ids[u]][cluster_ids[v]];
          all_edges[cluster_ids[u]][cluster_ids[v]] = aggregate_func(element, FloatFromWeightCCI(weight));
          //.push_back(std::make_tuple(cluster_ids[v], weight));
            }
      };
      vtx.mapOutNgh(j, map_f, false);
    }
  }

  return all_edges;
}


template<class G>
OffsetsEdges SeqComputeInterClusterEdgesSort(
    G& original_graph,
    const std::vector<gbbs::uintE>& cluster_ids,
    std::size_t num_compressed_vertices,
    const std::function<float(float, float)>& aggregate_func,
    const std::function<bool(gbbs::uintE, gbbs::uintE)>& is_valid_func, gbbs::uintE max_cluster_id) {
  // Retrieve all valid edges, mapped to cluster_ids
  auto inter_cluster_edges =
      SeqRetrieveInterClusterEdges(original_graph, cluster_ids, is_valid_func, max_cluster_id, aggregate_func);
  std::vector<gbbs::uintE> offsets(max_cluster_id + 2);
  for (std::size_t i = 0; i < inter_cluster_edges.size() + 1; i++) {
    if (i == 0) offsets[i] = 0;
    else offsets[i] = offsets[i - 1] + inter_cluster_edges[i - 1].size();
  }

  // Filter out unique edges into edges
  // This is done by iterating over the boundaries where edges differ,
  // and retrieving all of the same edges in one section. These edges
  // are combined using aggregate_func.
  auto num_filtered_mark_edges = offsets[inter_cluster_edges.size()];
  std::unique_ptr<std::tuple<gbbs::uintE, float>[]> edges(
      new std::tuple<gbbs::uintE, float>[num_filtered_mark_edges]);
  std::size_t idx = 0;
  for (std::size_t i = 0; i < inter_cluster_edges.size(); i++) {
    for (auto& x : inter_cluster_edges[i]) {
      edges[idx] = std::make_tuple(gbbs::uintE{x.first}, float{x.second});
      idx++;
    }
  }
  return OffsetsEdges{offsets, std::move(edges), num_filtered_mark_edges};
}


// This class encapsulates the data needed to compute and maintain the
// correlation clustering objective.
class SeqClusteringHelper {
 public:
  using ClusterId = gbbs::uintE;

  SeqClusteringHelper(gbbs::uintE num_nodes,
                   const ClustererConfig& clusterer_config,
                   const std::vector<std::vector<gbbs::uintE>>& clustering)
      : num_nodes_(num_nodes),
        cluster_ids_(num_nodes),
        cluster_sizes_(num_nodes, 0),
        clusterer_config_(clusterer_config),
        node_weights_(num_nodes, 1),
        cluster_weights_(num_nodes, 0) {
    SetClustering(clustering);
  }

  SeqClusteringHelper(gbbs::uintE num_nodes,
                   const ClustererConfig& clusterer_config,
                   std::vector<double> node_weights,
                   const std::vector<std::vector<gbbs::uintE>>& clustering)
      : num_nodes_(num_nodes),
        cluster_ids_(num_nodes),
        cluster_sizes_(num_nodes, 0),
        clusterer_config_(clusterer_config),
        node_weights_(std::move(node_weights)),
        cluster_weights_(num_nodes, 0) {
    SetClustering(clustering);
  }

  // Contains objective change, which includes:
  //  * A vector of tuples, indicating the objective change for the
  //    corresponding cluster id if a node is moved to said cluster.
  //  * The objective change of a node moving out of its current cluster
  struct ObjectiveChange {
    std::vector<std::tuple<ClusterId, double>> move_to_change;
    double move_from_change;
  };
  
  template<class G>
  std::tuple<SeqClusteringHelper::ClusterId, double> EfficientBestMove(
    G& graph,
    gbbs::uintE moving_node) {
      using W = typename G::weight_type;
  const auto& config = clusterer_config_.correlation_clusterer_config();
  const double offset = config.edge_weight_offset();

  // Class 2 edges where the endpoints are currently in different clusters.
  //EdgeSum class_2_currently_separate;
  // Class 1 edges where the endpoints are currently in the same cluster.
  EdgeSum class_1_currently_together;
  // Class 1 edges, grouped by the cluster that the non-moving node is in.
  absl::flat_hash_map<ClusterId, EdgeSum> class_1_together_after;

  double moving_nodes_weight = 0;
  const ClusterId node_cluster = cluster_ids_[moving_node];
  //cluster_moving_weights[node_cluster] += node_weights_[moving_node];
  moving_nodes_weight += node_weights_[moving_node];

  auto map_moving_node_neighbors = [&](gbbs::uintE u, gbbs::uintE neighbor,
                                       W w) {
    float weight = FloatFromWeightCCI(w);
    weight -= offset;
    const ClusterId neighbor_cluster = cluster_ids_[neighbor];
    if (moving_node != neighbor) {
      // Class 1 edge.
      if (node_cluster == neighbor_cluster) {
        class_1_currently_together.Add(weight);
      }
      class_1_together_after[neighbor_cluster].Add(weight);
    }
  };
  graph.get_vertex(moving_node)
      .mapOutNgh(moving_node, map_moving_node_neighbors, false);

 double change_in_objective = 0;

  double max_edges = 0;

  max_edges = node_weights_[moving_node] *
              (cluster_weights_[node_cluster] - node_weights_[moving_node]);
  change_in_objective -= class_1_currently_together.NetWeight(max_edges, config);

  std::pair<absl::optional<ClusterId>, double> best_move;
  best_move.first = absl::nullopt;
  best_move.second = change_in_objective;
  for (const auto& [cluster, data] : class_1_together_after) {
    max_edges = moving_nodes_weight * (cluster_weights_[cluster]);
    // Change in objective if we move the moving nodes to cluster i.
    double overall_change_in_objective =
        change_in_objective + data.NetWeight(max_edges, config);
    if (overall_change_in_objective > best_move.second ||
        (overall_change_in_objective == best_move.second &&
         cluster < best_move.first)) {
      best_move.first = cluster;
      best_move.second = overall_change_in_objective;
    }
  }

  auto move_id =
      best_move.first.has_value() ? best_move.first.value() : graph.n;
  std::tuple<ClusterId, double> best_move_tuple =
      std::make_tuple(move_id, best_move.second);

  return best_move_tuple;
}

  template<class G>
  bool AsyncMove(
    G& graph,
    gbbs::uintE moving_node) {
  auto best_move = EfficientBestMove(graph, moving_node);

  auto move_cluster_id = std::get<0>(best_move);
  auto current_cluster_id = ClusterIds()[moving_node];
  if (std::get<1>(best_move) <= 0) return false;

  cluster_sizes_[current_cluster_id]--;
  cluster_weights_[current_cluster_id] -= node_weights_[moving_node];

  if (move_cluster_id != num_nodes_) {
    cluster_sizes_[move_cluster_id]++;
    cluster_weights_[move_cluster_id] += node_weights_[moving_node];
    cluster_ids_[moving_node] = move_cluster_id;
    return true;
  }

  std::size_t i = 0;
  while(true) {
    if (cluster_sizes_[i] == 0) {
      cluster_sizes_[i] = 1;
      cluster_weights_[i] += node_weights_[moving_node];
      cluster_ids_[moving_node] = i;
      return true;
    }
    i++;
    i = i % num_nodes_;
  }
  return true;
}
  


  // Returns a tuple of:
  //  * The best cluster to move moving_node to according to the correlation
  //    clustering objective function. An id equal to the number of nodes in the
  //    graph means create a new cluster.
  //  * The change in objective function achieved by that move. May be positive
  //    or negative.
  std::tuple<ClusterId, double> BestMove(
      gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
      gbbs::uintE moving_node);

  const std::vector<ClusterId>& ClusterIds() const { return cluster_ids_; }

  const std::vector<ClusterId>& ClusterSizes() const { return cluster_sizes_; }

  const std::vector<double>& ClusterWeights() const { return cluster_weights_; }

  // Returns the weight of the given node, or 1.0 if it has not been set.
  double NodeWeight(gbbs::uintE id) const;

  // Initialize cluster_ids_ and cluster_sizes_ given an initial clustering.
  // If clustering is empty, initialize singleton clusters.
  void SetClustering(const std::vector<std::vector<gbbs::uintE>>& clustering);

  void ResetClustering(const std::vector<std::vector<gbbs::uintE>>& clustering);

 private:
  std::size_t num_nodes_;
  std::vector<ClusterId> cluster_ids_;
  std::vector<ClusterId> cluster_sizes_;
  ClustererConfig clusterer_config_;
  std::vector<double> node_weights_;
  std::vector<double> cluster_weights_;
};

// Given cluster ids and a graph, compress the graph such that the new
// vertices are the cluster ids and the edges are aggregated by sum.
// Self-loops preserve the total weight of the undirected edges in the clusters.
template<class G>
absl::StatusOr<GraphWithWeights> SeqCompressGraph(
    G& original_graph,
    const std::vector<gbbs::uintE>& cluster_ids, SeqClusteringHelper* helper) {
  // Obtain the number of vertices in the new graph
  gbbs::uintE max_cluster_id = 0;
  for (std::size_t i = 0; i < cluster_ids.size(); i++) {
    if (cluster_ids[i] != UINT_E_MAX && max_cluster_id < cluster_ids[i]) max_cluster_id = cluster_ids[i];
  }
  gbbs::uintE num_compressed_vertices = 1 + max_cluster_id;

  // Compute new inter cluster edges using sorting, allowing self-loops
  auto edge_aggregation_func = [](double w1, double w2) { return w1 + w2; };
  auto is_valid_func = [](ClusteringHelper::ClusterId a,
                          ClusteringHelper::ClusterId b) { return true; };

  OffsetsEdges offsets_edges = SeqComputeInterClusterEdgesSort(
      original_graph, cluster_ids, num_compressed_vertices,
      edge_aggregation_func, is_valid_func, max_cluster_id);
  std::vector<gbbs::uintE> offsets = offsets_edges.offsets;
  std::size_t num_edges = offsets_edges.num_edges;
  std::unique_ptr<std::tuple<gbbs::uintE, float>[]> edges =
      std::move(offsets_edges.edges);

  // Initialize new node weights
  std::vector<double> new_node_weights(num_compressed_vertices, 0);

  // Obtain cluster ids and node weights of all vertices
  for (std::size_t i = 0; i < original_graph.n; i++) {
    new_node_weights[cluster_ids[i]] += helper->NodeWeight(i);
  }

  return GraphWithWeights(SeqMakeGbbsGraph<float>(offsets, num_compressed_vertices,
                                               std::move(edges), num_edges),
                          new_node_weights);
}


}  // namespace in_memory
}  // namespace research_graph

#endif  // RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_CORRELATION_CLUSTERER_INTERNAL_H_