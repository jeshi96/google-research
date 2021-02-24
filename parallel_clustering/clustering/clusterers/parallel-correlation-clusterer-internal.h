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

#ifndef RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_PARALLEL_CORRELATION_CLUSTERER_INTERNAL_H_
#define RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_PARALLEL_CORRELATION_CLUSTERER_INTERNAL_H_

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


#include "absl/strings/str_cat.h"
#include "clustering/clusterers/correlation-clusterer-util.h"
#include "external/gbbs/gbbs/bridge.h"
#include "external/gbbs/gbbs/gbbs.h"
#include "external/gbbs/gbbs/macros.h"
#include "external/gbbs/pbbslib/random_shuffle.h"
#include "external/gbbs/pbbslib/sample_sort.h"
#include "external/gbbs/pbbslib/seq.h"
#include "external/gbbs/pbbslib/sequence_ops.h"
#include "external/gbbs/pbbslib/utilities.h"
#include "parallel/parallel-graph-utils.h"
#include "parallel/parallel-sequence-ops.h"
#include "external/gbbs/gbbs/pbbslib/sparse_table.h"
#include "external/gbbs/gbbs/pbbslib/sparse_additive_map.h"


namespace research_graph {
namespace in_memory {

float FloatFromWeightPCCI(float weight);
float FloatFromWeightPCCI(pbbslib::empty weight);


using ClusterId = gbbs::uintE;

// This class encapsulates the data needed to compute and maintain the
// correlation clustering objective.
class ClusteringHelper {
 public:
  using ClusterId = gbbs::uintE;

  ClusteringHelper(gbbs::uintE num_nodes,
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

  ClusteringHelper(gbbs::uintE num_nodes,
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

  // Constructor for testing purposes, to outright set cluster_ids_ and
  // cluster_sizes_.
  ClusteringHelper(size_t num_nodes, std::vector<ClusterId> cluster_ids,
                   std::vector<ClusterId> cluster_sizes,
                   std::vector<double> cluster_weights,
                   const ClustererConfig& clusterer_config,
                   std::vector<double> node_weights)
      : num_nodes_(num_nodes),
        cluster_ids_(std::move(cluster_ids)),
        cluster_sizes_(std::move(cluster_sizes)),
        clusterer_config_(clusterer_config),
        node_weights_(std::move(node_weights)),
        cluster_weights_(cluster_weights) {}

  // Contains objective change, which includes:
  //  * A vector of tuples, indicating the objective change for the
  //    corresponding cluster id if a node is moved to said cluster.
  //  * The objective change of a node moving out of its current cluster
  struct ObjectiveChange {
    std::vector<std::tuple<ClusterId, double>> move_to_change;
    double move_from_change;
  };

  // Moves node i from its current cluster to a new cluster moves[i].
  // If moves[i] == null optional, then the corresponding node will not be
  // moved. A move to the number of nodes in the graph means that a new cluster
  // is created. The size of moves should be equal to num_nodes_.
  // Returns an array where the entry is true if the cluster corresponding to
  // the index was modified, and false if the cluster corresponding to the
  // index was not modified. Nodes may not necessarily move if the best move
  // provided is to stay in their existing cluster.
/*
  std::unique_ptr<bool[]> MoveNodesToCluster(
      std::vector<absl::optional<ClusterId>>& moves,
      std::vector<double>& moves_obj,
      gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph);*/

std::unique_ptr<bool[]> MoveNodesToCluster(
      const std::vector<absl::optional<ClusterId>>& moves);

  // Returns a tuple of:
  //  * The best cluster to move all of the nodes in moving_nodes to according
  //    to the correlation clustering objective function. An id equal to the
  //    number of nodes in the graph means create a new cluster.
  //  * The change in objective function achieved by that move. May be positive
  //    or negative.
  /*std::tuple<ClusteringHelper::ClusterId, double> BestMove(
      gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
      const std::vector<gbbs::uintE>& moving_nodes);*/
  
  template<class Graph>
  std::tuple<ClusteringHelper::ClusterId, double> EfficientBestMove(
    Graph& graph,
    gbbs::uintE moving_node) {
  using W = typename Graph::weight_type;
  const auto& config = clusterer_config_.correlation_clusterer_config();
  const double offset = config.edge_weight_offset();

  auto deg = graph.get_vertex(moving_node).getOutDegree();
  assert(deg < graph.n);

  if (deg <= 1000) {

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
    float weight = FloatFromWeightPCCI(w);
    weight -= offset;
    const ClusterId neighbor_cluster = cluster_ids_[neighbor];
    /*if (moving_node == neighbor) {
      // Class 2 edge.
      if (node_cluster != neighbor_cluster) {
        class_2_currently_separate.Add(weight);
      }
    } else*/ if (moving_node != neighbor) {
      // Class 1 edge.
      if (node_cluster == neighbor_cluster) {
        class_1_currently_together.Add(weight);
      }
      class_1_together_after[neighbor_cluster].Add(weight);
    }
  };
  graph.get_vertex(moving_node)
      .mapOutNgh(moving_node, map_moving_node_neighbors, false);
  //class_2_currently_separate.RemoveDoubleCounting();
  // Now cluster_moving_weights is correct and class_2_currently_separate,
  // class_1_currently_together, and class_1_by_cluster are ready to call
  // NetWeight().

 double change_in_objective = 0;

  double max_edges = 0;
  //change_in_objective += class_2_currently_separate.NetWeight(max_edges, config);

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

   auto curr_together_seq = gbbs::sequence<double>(deg, [](std::size_t i){return double{0};});

  auto together_after_table = pbbslib::sparse_additive_map(deg, std::make_tuple(UINT_E_MAX, double{0}));

  // Class 1 edges, grouped by the cluster that the non-moving node is in.
  absl::flat_hash_map<ClusterId, EdgeSum> class_1_together_after;

  double moving_nodes_weight = 0;
  const ClusterId node_cluster = cluster_ids_[moving_node];
  //cluster_moving_weights[node_cluster] += node_weights_[moving_node];
  moving_nodes_weight += node_weights_[moving_node];

  auto map_moving_node_neighbors = [&](const gbbs::uintE u_id, const gbbs::uintE neighbor,
    W w, const gbbs::uintE j){
      float weight = FloatFromWeightPCCI(w);
      weight -= offset;
      const ClusterId neighbor_cluster = cluster_ids_[neighbor];
      if (moving_node != neighbor) {
        // Class 1 edge.
        if (node_cluster == neighbor_cluster) {
          curr_together_seq[j] = weight;
          //class_1_currently_together.Add(weight);
        }
        together_after_table.insert({gbbs::uintE{neighbor_cluster}, double{weight}});
        //class_1_together_after[neighbor_cluster].Add(weight);
      }
    };
    graph.get_vertex(moving_node)
      .mapOutNghWithIndex(moving_node, map_moving_node_neighbors);

 double change_in_objective = 0;

  double max_edges = 0;
  //change_in_objective += class_2_currently_separate.NetWeight(max_edges, config);

  max_edges = node_weights_[moving_node] *
              (cluster_weights_[node_cluster] - node_weights_[moving_node]);
  double curr_together = pbbslib::reduce_add(curr_together_seq) - config.resolution() * max_edges;
  change_in_objective -= curr_together;

  auto together_after_entries = together_after_table.entries();
  using M = std::tuple<ClusterId, double>;
  auto best_move_seq = gbbs::sequence<M>(together_after_entries.size());

  pbbs::parallel_for(0, together_after_entries.size(), [&](std::size_t i) {
    auto cluster = std::get<0>(together_after_entries[i]);
    auto data = std::get<1>(together_after_entries[i]);
    max_edges = moving_nodes_weight * (cluster_weights_[cluster]);
    data -= config.resolution() * max_edges;
    double overall_change_in_objective = change_in_objective + data;
    best_move_seq[i] = std::make_tuple(ClusterId{cluster}, overall_change_in_objective);
  });

  // Retrieve max of best_move_seq
  auto f_max = [](const M& a, const M& b){
    if (std::get<1>(a) == std::get<1>(b)) {
      if (std::get<0>(a) < std::get<0>(b)) return a;
      return b;
    }
    if (std::get<1>(a) > std::get<1>(b)) return a;
    return b;
  };
  auto max_monoid = pbbs::make_monoid(f_max, std::make_tuple(ClusterId{0}, double{0}));
  auto max_move = pbbs::reduce(best_move_seq.slice(), max_monoid);

  if (change_in_objective >= std::get<1>(max_move)) {
    std::tuple<ClusterId, double> best_move_tuple =
      std::make_tuple(graph.n, change_in_objective);
    return best_move_tuple;
  }
  return max_move;
}
  
template<class Graph>
  bool AsyncMove(
    Graph& graph,
    gbbs::uintE moving_node) {
  auto best_move = EfficientBestMove(graph, moving_node);

  auto move_cluster_id = std::get<0>(best_move);
  auto current_cluster_id = ClusterIds()[moving_node];
  if (std::get<1>(best_move) <= 0) return false;
  /*if (move_cluster_id < graph.n &&
      ClusterSizes()[move_cluster_id] == 1 &&
      ClusterSizes()[current_cluster_id] == 1 &&
      current_cluster_id >= move_cluster_id) {
    return false;
  }*/
  pbbs::write_add(&cluster_sizes_[current_cluster_id], -1);
  pbbs::write_add(&cluster_weights_[current_cluster_id], -1 * node_weights_[moving_node]);
  //cluster_sizes_[current_cluster_id]--;
  //cluster_weights_[current_cluster_id] -= node_weights_[moving_node];

  if (move_cluster_id != num_nodes_) {
    pbbs::write_add(&cluster_sizes_[move_cluster_id], 1);
    pbbs::write_add(&cluster_weights_[move_cluster_id], node_weights_[moving_node]);
    cluster_ids_[moving_node] = move_cluster_id;
    //cluster_sizes_[move_cluster_id]++;
    //cluster_weights_[move_cluster_id] += node_weights_[moving_node];
    return true;
  }

  std::size_t i = 0;
  while(true) {
    if (cluster_sizes_[i] == 0) {
      if (pbbslib::CAS<ClusterId>(&cluster_sizes_[i], 0, 1)) {
        pbbs::write_add(&cluster_weights_[i], node_weights_[moving_node]);
        cluster_ids_[moving_node] = i;
        return true;
      }
    }
    i++;
    i = i % num_nodes_;
  }
  return true;
}

  template<class Graph>
  bool AsyncMove(
    Graph& graph,
    const std::vector<gbbs::uintE>& moving_nodes) {
  if (moving_nodes.size() == 0) return false;
  else if (moving_nodes.size() == 1) return AsyncMove(graph, moving_nodes[0]);
  
  std::tuple<ClusteringHelper::ClusterId, double> best_move = BestMove(graph, moving_nodes);
  auto move_cluster_id = std::get<0>(best_move);
  if (std::get<1>(best_move) <= 0) return false;
  auto curr_cluster_id = ClusterIds()[moving_nodes[0]];
  pbbs::write_add(&cluster_sizes_[curr_cluster_id], -1 * moving_nodes.size());
  auto weights_seq = pbbs::delayed_seq<float>(moving_nodes.size(), [&](std::size_t i) {
    return node_weights_[moving_nodes[i]];
  });
  auto total_weight = pbbslib::reduce_add(weights_seq);
  pbbs::write_add(&cluster_weights_[curr_cluster_id], -1 * total_weight);

  if (move_cluster_id != num_nodes_) {
    pbbs::write_add(&cluster_sizes_[move_cluster_id], moving_nodes.size());
    pbbs::write_add(&cluster_weights_[move_cluster_id], total_weight);
    pbbs::parallel_for(0, moving_nodes.size(), [&](std::size_t i) {
      cluster_ids_[moving_nodes[i]] = move_cluster_id;
    });
    return true;
  }
  std::size_t i = 0;
  while(true) {
    if (cluster_sizes_[i] == 0) {
      if (pbbslib::CAS<ClusterId>(&cluster_sizes_[i], 0, moving_nodes.size())) {
        pbbs::write_add(&cluster_weights_[i], total_weight);
        pbbs::parallel_for(0, moving_nodes.size(), [&](std::size_t j) {
          cluster_ids_[moving_nodes[j]] = i;
        });
        return true;
      }
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
  /*std::tuple<ClusterId, double> BestMove(
      gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
      InMemoryClusterer::NodeId moving_node);*/

  // Compute the objective of the current clustering
  template<class Graph>
  double ComputeObjective(
      Graph& graph){
  using W = typename Graph::weight_type;
  const auto& config = clusterer_config_.correlation_clusterer_config();
  std::vector<double> shifted_edge_weight(graph.n);

  // Compute cluster statistics contributions of each vertex
  pbbs::parallel_for(0, graph.n, [&](std::size_t i) {
    gbbs::uintE cluster_id_i = cluster_ids_[i];
    auto add_m = pbbslib::addm<double>();

    auto intra_cluster_sum_map_f = [&](gbbs::uintE u, gbbs::uintE v,
                                       W weight) -> double{
      // This assumes that the graph is undirected, and self-loops are counted
      // as half of the weight.
      if (cluster_id_i == cluster_ids_[v])
        return (FloatFromWeightPCCI(weight) - config.edge_weight_offset()) / 2;
      return 0;
    };
    auto vtx = graph.get_vertex(i);
    shifted_edge_weight[i] = vtx.template reduceOutNgh<double>(
        i, intra_cluster_sum_map_f, add_m);
  });
  double objective =
      parallel::ReduceAdd(absl::Span<const double>(shifted_edge_weight));

  auto resolution_seq = pbbs::delayed_seq<double>(graph.n, [&](std::size_t i) {
    auto cluster_weight = cluster_weights_[cluster_ids_[i]];
    return node_weights_[i] * (cluster_weight - node_weights_[i]);
  });
  objective -= config.resolution() * pbbslib::reduce_add(resolution_seq) / 2;

  return objective;
}

  
  template <class Graph>
  double ComputeDisagreementObjective(
      Graph& graph) {
  // They minimize sum of edges in b/w clusters (edge weight - res * weight i * weight j) + non-edges in cluster (res * w_i w_j)
  const auto& config = clusterer_config_.correlation_clusterer_config();
  std::vector<double> shifted_edge_weight(graph.n);
  using W = typename Graph::weight_type;

  // Compute cluster statistics contributions of each vertex
  pbbs::parallel_for(0, graph.n, [&](std::size_t i) {
    gbbs::uintE cluster_id_i = cluster_ids_[i];
    auto add_m = pbbslib::addm<double>();

    auto intra_cluster_sum_map_f = [&](gbbs::uintE u, gbbs::uintE v,
                                      W w) -> double{
      float weight = FloatFromWeightPCCI(w);
      // This assumes that the graph is undirected, and self-loops are counted
      // as half of the weight.
      if (cluster_id_i != cluster_ids_[v])
        return ((weight - config.edge_weight_offset()) / 2) - config.resolution() * node_weights_[u] * node_weights_[v];
      else
        return (-1 * config.resolution() * node_weights_[u] * node_weights_[v]) / 2;
      return 0;
    };
    auto vtx = graph.get_vertex(i);
    shifted_edge_weight[i] = vtx.template reduceOutNgh<double>(
        i, intra_cluster_sum_map_f, add_m);
  });
  double objective =
      parallel::ReduceAdd(absl::Span<const double>(shifted_edge_weight));

  auto resolution_seq = pbbs::delayed_seq<double>(graph.n, [&](std::size_t i) {
    auto cluster_weight = cluster_weights_[cluster_ids_[i]];
    return node_weights_[i] * (cluster_weight);// - node_weights_[i]);
  });
  objective += config.resolution() * pbbslib::reduce_add(resolution_seq) / 2;
  // Note that here, we're counting non-existent self-loops inside a cluster as "non-edges"
  return objective;
}


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

// Holds a GBBS graph and a corresponding node weights
struct GraphWithWeights {
  GraphWithWeights() {}
  GraphWithWeights(
      std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>
          graph_,
      std::vector<double> node_weights_)
      : graph(std::move(graph_)), node_weights(std::move(node_weights_)) {}
  std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>
      graph;
  std::vector<double> node_weights;
};

// Given cluster ids and a graph, compress the graph such that the new
// vertices are the cluster ids and the edges are aggregated by sum.
// Self-loops preserve the total weight of the undirected edges in the clusters.
template <class Graph>
absl::StatusOr<GraphWithWeights> CompressGraph(
    Graph& original_graph,
    const std::vector<gbbs::uintE>& cluster_ids, ClusteringHelper* helper){
  // Obtain the number of vertices in the new graph
  auto get_cluster_ids = [&](size_t i) { return cluster_ids[i]; };
  auto seq_cluster_ids =
      pbbs::delayed_seq<gbbs::uintE>(cluster_ids.size(), get_cluster_ids);
  gbbs::uintE num_compressed_vertices =
      1 + pbbslib::reduce_max(seq_cluster_ids);

  // Compute new inter cluster edges using sorting, allowing self-loops
  auto edge_aggregation_func = [](double w1, double w2) { return w1 + w2; };
  auto is_valid_func = [](ClusteringHelper::ClusterId a,
                          ClusteringHelper::ClusterId b) { return true; };

  OffsetsEdges offsets_edges = ComputeInterClusterEdgesSort(
      original_graph, cluster_ids, num_compressed_vertices,
      edge_aggregation_func, is_valid_func);
  std::vector<gbbs::uintE> offsets = offsets_edges.offsets;
  std::size_t num_edges = offsets_edges.num_edges;
  std::unique_ptr<std::tuple<gbbs::uintE, float>[]> edges =
      std::move(offsets_edges.edges);

  // Obtain cluster ids and node weights of all vertices
  std::vector<std::tuple<ClusterId, double>> node_weights(original_graph.n);
  pbbs::parallel_for(0, original_graph.n, [&](std::size_t i) {
    node_weights[i] = std::make_tuple(cluster_ids[i], helper->NodeWeight(i));
  });

  // Initialize new node weights
  std::vector<double> new_node_weights(num_compressed_vertices, 0);

  // Sort weights of neighbors by cluster id
  auto node_weights_sort = research_graph::parallel::ParallelSampleSort<
      std::tuple<ClusterId, double>>(
      absl::Span<std::tuple<ClusterId, double>>(node_weights.data(),
                                                node_weights.size()),
      [&](std::tuple<ClusterId, double> a, std::tuple<ClusterId, double> b) {
        return std::get<0>(a) < std::get<0>(b);
      });

  // Obtain the boundary indices where cluster ids differ
  std::vector<gbbs::uintE> mark_node_weights =
      parallel::GetBoundaryIndices<gbbs::uintE>(
          node_weights_sort.size(),
          [&node_weights_sort](std::size_t i, std::size_t j) {
            return std::get<0>(node_weights_sort[i]) ==
                   std::get<0>(node_weights_sort[j]);
          });
  std::size_t num_mark_node_weights = mark_node_weights.size() - 1;


  // Reset helper to singleton clusters, with appropriate node weights
  pbbs::parallel_for(0, num_mark_node_weights, [&](std::size_t i) {
    gbbs::uintE start_id_index = mark_node_weights[i];
    gbbs::uintE end_id_index = mark_node_weights[i + 1];
    auto node_weight =
        research_graph::parallel::Reduce<std::tuple<ClusterId, double>>(
            absl::Span<const std::tuple<ClusterId, double>>(
                node_weights_sort.begin() + start_id_index,
                end_id_index - start_id_index),
            [&](std::tuple<ClusterId, double> a,
                std::tuple<ClusterId, double> b) {
              return std::make_tuple(std::get<0>(a),
                                     std::get<1>(a) + std::get<1>(b));
            },
            std::make_tuple(std::get<0>(node_weights[start_id_index]),
                            double{0}));
    new_node_weights[std::get<0>(node_weight)] = std::get<1>(node_weight);
  });

  return GraphWithWeights(MakeGbbsGraph<float>(offsets, num_compressed_vertices,
                                               std::move(edges), num_edges),
                          new_node_weights);
}


}  // namespace in_memory
}  // namespace research_graph

#endif  // RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_PARALLEL_CORRELATION_CLUSTERER_INTERNAL_H_