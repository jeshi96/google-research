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

#include "clustering/clusterers/parallel-correlation-clusterer-internal.h"

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

#include "external/gbbs/benchmarks/Connectivity/WorkEfficientSDB14/Connectivity.h"

namespace research_graph {
namespace in_memory {

using NodeId = InMemoryClusterer::NodeId;
using ClusterId = ClusteringHelper::ClusterId;

void ClusteringHelper::ResetClustering(
  const InMemoryClusterer::Clustering& clustering) {
  pbbs::parallel_for(0, num_nodes_, [&](std::size_t i) {
      cluster_weights_[i] = 0;
      cluster_sizes_[i] = 0;
      
  });
  SetClustering(clustering);
}

void ClusteringHelper::SetClustering(
    const InMemoryClusterer::Clustering& clustering) {
  if (clustering.empty()) {
    pbbs::parallel_for(0, num_nodes_, [&](std::size_t i) {
      cluster_sizes_[i] = 1;
      cluster_ids_[i] = i;
      cluster_weights_[i] = node_weights_[i];
    });
  } else {
    pbbs::parallel_for(0, clustering.size(), [&](std::size_t i) {
      cluster_sizes_[i] = clustering[i].size();
      for (auto j : clustering[i]) {
        cluster_ids_[j] = i;
        cluster_weights_[i] += node_weights_[j];
      }
    });
  }
}

double ClusteringHelper::NodeWeight(NodeId id) const {
  return id < node_weights_.size() ? node_weights_[id] : 1.0;
}

double ClusteringHelper::ComputeObjective(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph) {
  const auto& config = clusterer_config_.correlation_clusterer_config();
  std::vector<double> shifted_edge_weight(graph.n);

  // Compute cluster statistics contributions of each vertex
  pbbs::parallel_for(0, graph.n, [&](std::size_t i) {
    gbbs::uintE cluster_id_i = cluster_ids_[i];
    auto add_m = pbbslib::addm<double>();

    auto intra_cluster_sum_map_f = [&](gbbs::uintE u, gbbs::uintE v,
                                       float weight) -> double {
      // This assumes that the graph is undirected, and self-loops are counted
      // as half of the weight.
      if (cluster_id_i == cluster_ids_[v])
        return (weight - config.edge_weight_offset()) / 2;
      return 0;
    };
    shifted_edge_weight[i] = graph.get_vertex(i).reduceOutNgh<double>(
        i, intra_cluster_sum_map_f, add_m);
  });
  double objective =
      parallel::ReduceAdd(absl::Span<const double>(shifted_edge_weight));

  auto resolution_seq = pbbs::delayed_seq<double>(graph.n, [&](std::size_t i) {
    auto cluster_weight = cluster_weights_[cluster_ids_[i]];
    return node_weights_[i] * (cluster_weight);// - node_weights_[i]);
  });
  objective -= config.resolution() * pbbslib::reduce_add(resolution_seq) / 2;

  return objective;
}

std::unique_ptr<bool[]> ClusteringHelper::MoveNodesToCluster(
  std::vector<absl::optional<ClusterId>>& moves,
  std::vector<double>& moves_obj,
  gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph) {
  const auto& config = clusterer_config_.correlation_clusterer_config();
  const double offset = config.edge_weight_offset();
  const double resolution = config.resolution();
  // Sort moves by moves_obj
  using M = std::tuple<gbbs::uintE, absl::optional<ClusterId>, double>;
  auto get_moves_func = [&](std::size_t i) {
    return std::make_tuple(gbbs::uintE{i}, moves[i], moves_obj[i]);
  };
  auto moves_sort = pbbs::sample_sort(
      pbbs::delayed_seq<M>(moves.size(), get_moves_func),
      [&](M a, M b) { return std::get<2>(a) > std::get<2>(b); }, true);
  // Then, compute prefix sum type thing of what total obj to that point would be
  // To do this, first, for each obj at vtx i, and for each prior vtx j,
  // if i and j are moving to the same cluster, then add val depending on if there's
  // an edge or not; also, if i and j were previously in the same cluster, then
  // we must have thought that i and j will be separated, so we would've subtracted
  // val; we must add back in val
  // Then do prefix sum
  pbbs::parallel_for(0, moves_sort.size(), [&](std::size_t i) {
    auto vtx_idx = std::get<0>(moves_sort[i]);
    if (std::get<1>(moves_sort[i]).has_value()) {
      auto move_id = std::get<1>(moves_sort[i]).value();
      double val_total = 0;
      std::unordered_map<gbbs::uintE, float> vtx_nbhrs;
      auto vtx = current_graph->get_vertex(vtx_idx);
      for (std::size_t k = 0; k < vtx.getOutDegree(); k++) {
        vtx_nbhrs.insert(std::make_pair(vtx.getOutNeighbor(k), std::get<1>((vtx.getOutNeighbors())[k])));
      }
      for (std::size_t j = 0; j < i; j++) {
        auto vtx_idx2 = std::get<0>(moves_sort[j]);
        if (!std::get<1>(moves_sort[j]).has_value()) break;
        auto move_id2 = std::get<1>(moves_sort[j]).value();
        // now we know both i and j are moving
      // if they were previously in different clusters, and now are moving to the same
      // cluster -- add twice (w_uv - edge_weight_offset - resolution k_u k_v) if edge, (-resolution k_u k_v) otherwise

      // if they were previously in the same cluster, and are now moving to diff clusters, do nothing
      // if they were previously in the same cluster, and are now moving to the same cluster,
      // add twice the val above
 
      // if they are moving to the same place
        if (move_id == move_id2) {
          double val = -1 * resolution * node_weights_[vtx_idx] * node_weights_[vtx_idx2];
          auto find = vtx_nbhrs.find(vtx_idx2);
          if (find != vtx_nbhrs.end()) {
            val += find->second - offset;
          }
          val_total += val;
        }
        // if one thought they were moving to the cluster of the other, but now they're not
      // if they were previously in the same cluster
      //if (cluster_ids_[vtx_idx] == cluster_ids_[vtx_idx2])
      }
      moves_sort[i] = std::make_tuple(vtx_idx, std::get<1>(moves_sort[i]), val_total + std::get<2>(moves_sort[i]));
    }
  });
  auto f = [](const M& a, const M& b){
    return std::make_tuple(std::get<0>(b), std::get<1>(b), std::get<2>(a) + std::get<2>(b));
  };
  auto mon = pbbslib::make_monoid(f, std::make_tuple(gbbs::uintE{0}, 0, double{0}));
  auto all = pbbs::scan_inplace(moves_sort.slice(), mon);
  // Find the max move
  std::size_t max_idx = moves_sort.size();
  double mm = std::get<2>(all);
  for (std::size_t i = 0; i < moves_sort.size(); i++) {
    if (std::get<2>(moves_sort[i]) > mm) {
      mm = std::get<2>(moves_sort[i]);
      max_idx = i;
    }
  }
  if (max_idx == moves_sort.size()) {
    moves_sort.clear();
    return MoveNodesToCluster(moves);
  }
  pbbs::parallel_for(max_idx + 1, moves_sort.size(), [&](std::size_t i){
    moves[std::get<0>(moves_sort[i])] = absl::optional<ClusterId>();
  });
  moves_sort.clear();
  return MoveNodesToCluster(moves);
  // Then, null out any moves after that -- and call MoveNodesToCluster on remaining
}

std::unique_ptr<bool[]> ClusteringHelper::MoveNodesToCluster(
    const std::vector<absl::optional<ClusterId>>& moves) {
  auto modified_cluster = absl::make_unique<bool[]>(num_nodes_);
  pbbs::parallel_for(0, num_nodes_,
                     [&](std::size_t i) { modified_cluster[i] = false; });

  // We must update cluster_sizes_ and assign new cluster ids to vertices
  // that want to form a new cluster
  // Obtain all nodes that are moving clusters
  auto get_moving_nodes = [&](size_t i) { return i; };
  auto moving_nodes = pbbs::filter(
      pbbs::delayed_seq<gbbs::uintE>(num_nodes_, get_moving_nodes),
      [&](gbbs::uintE node) -> bool {
        return moves[node].has_value() && moves[node] != cluster_ids_[node];
      },
      pbbs::no_flag);

  if (moving_nodes.empty()) return modified_cluster;

  // Sort moving nodes by original cluster id
  auto sorted_moving_nodes = pbbs::sample_sort(
      moving_nodes,
      [&](gbbs::uintE a, gbbs::uintE b) {
        return cluster_ids_[a] < cluster_ids_[b];
      },
      true);

  // The number of nodes moving out of clusters is given by the boundaries
  // where nodes differ by cluster id
  std::vector<gbbs::uintE> mark_moving_nodes =
      parallel::GetBoundaryIndices<gbbs::uintE>(
          sorted_moving_nodes.size(), [&](std::size_t i, std::size_t j) {
            return cluster_ids_[sorted_moving_nodes[i]] ==
                   cluster_ids_[sorted_moving_nodes[j]];
          });
  std::size_t num_mark_moving_nodes = mark_moving_nodes.size() - 1;

  // Subtract these boundary sizes from cluster_sizes_ in parallel
  pbbs::parallel_for(0, num_mark_moving_nodes, [&](std::size_t i) {
    gbbs::uintE start_id_index = mark_moving_nodes[i];
    gbbs::uintE end_id_index = mark_moving_nodes[i + 1];
    auto prev_id = cluster_ids_[sorted_moving_nodes[start_id_index]];
    cluster_sizes_[prev_id] -= (end_id_index - start_id_index);
    modified_cluster[prev_id] = true;
    for (std::size_t j = start_id_index; j < end_id_index; j++) {
      cluster_weights_[prev_id] -= node_weights_[sorted_moving_nodes[j]];
    }
  });

  // Re-sort moving nodes by new cluster id
  auto resorted_moving_nodes = pbbs::sample_sort(
      moving_nodes,
      [&](gbbs::uintE a, gbbs::uintE b) { return moves[a] < moves[b]; }, true);

  // The number of nodes moving into clusters is given by the boundaries
  // where nodes differ by cluster id
  std::vector<gbbs::uintE> remark_moving_nodes =
      parallel::GetBoundaryIndices<gbbs::uintE>(
          resorted_moving_nodes.size(),
          [&resorted_moving_nodes, &moves](std::size_t i, std::size_t j) {
            return moves[resorted_moving_nodes[i]] ==
                   moves[resorted_moving_nodes[j]];
          });
  std::size_t num_remark_moving_nodes = remark_moving_nodes.size() - 1;

  // Add these boundary sizes to cluster_sizes_ in parallel, excepting
  // those vertices that are forming new clusters
  // Also, excepting those vertices that are forming new clusters, update
  // cluster_ids_
  pbbs::parallel_for(0, num_remark_moving_nodes, [&](std::size_t i) {
    gbbs::uintE start_id_index = remark_moving_nodes[i];
    gbbs::uintE end_id_index = remark_moving_nodes[i + 1];
    auto move_id = moves[resorted_moving_nodes[start_id_index]].value();
    if (move_id != num_nodes_) {
      cluster_sizes_[move_id] += (end_id_index - start_id_index);
      modified_cluster[move_id] = true;
      for (std::size_t j = start_id_index; j < end_id_index; j++) {
        cluster_ids_[resorted_moving_nodes[j]] = move_id;
        cluster_weights_[move_id] += node_weights_[resorted_moving_nodes[j]];
      }
    }
  });

  // If there are vertices forming new clusters
  if (moves[resorted_moving_nodes[moving_nodes.size() - 1]].value() ==
      num_nodes_) {
    // Filter out cluster ids of empty clusters, so that these ids can be
    // reused for vertices forming new clusters. This is an optimization
    // so that cluster ids do not grow arbitrarily large, when assigning
    // new cluster ids.
    auto get_zero_clusters = [&](std::size_t i) { return i; };
    auto seq_zero_clusters =
        pbbs::delayed_seq<gbbs::uintE>(num_nodes_, get_zero_clusters);
    auto zero_clusters = pbbs::filter(
        seq_zero_clusters,
        [&](gbbs::uintE id) -> bool { return cluster_sizes_[id] == 0; },
        pbbs::no_flag);

    // Indexing into these cluster ids gives the new cluster ids for new
    // clusters; update cluster_ids_ and cluster_sizes_ appropriately
    gbbs::uintE start_id_index =
        remark_moving_nodes[num_remark_moving_nodes - 1];
    gbbs::uintE end_id_index = remark_moving_nodes[num_remark_moving_nodes];
    pbbs::parallel_for(start_id_index, end_id_index, [&](std::size_t i) {
      auto cluster_id = zero_clusters[i - start_id_index];
      cluster_ids_[resorted_moving_nodes[i]] = cluster_id;
      cluster_sizes_[cluster_id] = 1;
      modified_cluster[cluster_id] = true;
      cluster_weights_[cluster_id] = node_weights_[resorted_moving_nodes[i]];
    });
  }

  return modified_cluster;
}

// Just don't care about the atomics here tbh
bool ClusteringHelper::AsyncMove(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
    NodeId moving_node) {
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

// assume all moving nodes are in the same cluster
bool ClusteringHelper::AsyncMove(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
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


std::tuple<ClusteringHelper::ClusterId, double> ClusteringHelper::EfficientBestMove(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
    NodeId moving_node) {
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
                                       double weight) {
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

std::tuple<ClusteringHelper::ClusterId, double> ClusteringHelper::BestMove(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
    NodeId moving_node) {
  const auto& config = clusterer_config_.correlation_clusterer_config();
  const double offset = config.edge_weight_offset();

  // Weight of nodes in each cluster that are moving.
  absl::flat_hash_map<ClusterId, double> cluster_moving_weights;
  // Class 2 edges where the endpoints are currently in different clusters.
  EdgeSum class_2_currently_separate;
  // Class 1 edges where the endpoints are currently in the same cluster.
  EdgeSum class_1_currently_together;
  // Class 1 edges, grouped by the cluster that the non-moving node is in.
  absl::flat_hash_map<ClusterId, EdgeSum> class_1_together_after;

  double moving_nodes_weight = 0;
  const ClusterId node_cluster = cluster_ids_[moving_node];
  cluster_moving_weights[node_cluster] += node_weights_[moving_node];
  moving_nodes_weight += node_weights_[moving_node];
  auto map_moving_node_neighbors = [&](gbbs::uintE u, gbbs::uintE neighbor,
                                       double weight) {
    weight -= offset;
    const ClusterId neighbor_cluster = cluster_ids_[neighbor];
    if (moving_node == neighbor) {
      // Class 2 edge.
      if (node_cluster != neighbor_cluster) {
        class_2_currently_separate.Add(weight);
      }
    } else {
      // Class 1 edge.
      if (node_cluster == neighbor_cluster) {
        class_1_currently_together.Add(weight);
      }
      class_1_together_after[neighbor_cluster].Add(weight);
    }
  };
  graph.get_vertex(moving_node)
      .mapOutNgh(moving_node, map_moving_node_neighbors, false);
  class_2_currently_separate.RemoveDoubleCounting();
  // Now cluster_moving_weights is correct and class_2_currently_separate,
  // class_1_currently_together, and class_1_by_cluster are ready to call
  // NetWeight().

  std::function<double(ClusterId)> get_cluster_weight = [&](ClusterId cluster) {
    return cluster_weights_[cluster];
  };
  auto best_move =
      BestMoveFromStats(config, get_cluster_weight, moving_nodes_weight,
                        cluster_moving_weights, class_2_currently_separate,
                        class_1_currently_together, class_1_together_after);

  auto move_id =
      best_move.first.has_value() ? best_move.first.value() : graph.n;
  std::tuple<ClusterId, double> best_move_tuple =
      std::make_tuple(move_id, best_move.second);

  return best_move_tuple;
}

std::tuple<ClusteringHelper::ClusterId, double> ClusteringHelper::BestMove(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
    const std::vector<gbbs::uintE>& moving_nodes) {
  const auto& config = clusterer_config_.correlation_clusterer_config();
  const double offset = config.edge_weight_offset();

  std::vector<bool> flat_moving_nodes(graph.n, false);
  for (size_t i = 0; i < moving_nodes.size(); i++) {
    flat_moving_nodes[moving_nodes[i]] = true;
  }

  // Weight of nodes in each cluster that are moving.
  absl::flat_hash_map<ClusterId, double> cluster_moving_weights;
  // Class 2 edges where the endpoints are currently in different clusters.
  EdgeSum class_2_currently_separate;
  // Class 1 edges where the endpoints are currently in the same cluster.
  EdgeSum class_1_currently_together;
  // Class 1 edges, grouped by the cluster that the non-moving node is in.
  absl::flat_hash_map<ClusterId, EdgeSum> class_1_together_after;

  double moving_nodes_weight = 0;
  for (const auto& node : moving_nodes) {
    const ClusterId node_cluster = cluster_ids_[node];
    cluster_moving_weights[node_cluster] += node_weights_[node];
    moving_nodes_weight += node_weights_[node];
    auto map_moving_node_neighbors = [&](gbbs::uintE u, gbbs::uintE neighbor,
                                         float weight) {
      weight -= offset;
      const ClusterId neighbor_cluster = cluster_ids_[neighbor];
      if (flat_moving_nodes[neighbor]) {
        // Class 2 edge.
        if (node_cluster != neighbor_cluster) {
          class_2_currently_separate.Add(weight);
        }
      } else {
        // Class 1 edge.
        if (node_cluster == neighbor_cluster) {
          class_1_currently_together.Add(weight);
        }
        class_1_together_after[neighbor_cluster].Add(weight);
      }
    };
    graph.get_vertex(node).mapOutNgh(node, map_moving_node_neighbors, false);
  }
  class_2_currently_separate.RemoveDoubleCounting();
  // Now cluster_moving_weights is correct and class_2_currently_separate,
  // class_1_currently_together, and class_1_by_cluster are ready to call
  // NetWeight().

  std::function<double(ClusterId)> get_cluster_weight = [&](ClusterId cluster) {
    return cluster_weights_[cluster];
  };
  auto best_move =
      BestMoveFromStats(config, get_cluster_weight, moving_nodes_weight,
                        cluster_moving_weights, class_2_currently_separate,
                        class_1_currently_together, class_1_together_after);

  auto move_id =
      best_move.first.has_value() ? best_move.first.value() : graph.n;
  std::tuple<ClusterId, double> best_move_tuple =
      std::make_tuple(move_id, best_move.second);

  return best_move_tuple;
}

absl::StatusOr<GraphWithWeights> CompressGraph(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& original_graph,
    const std::vector<gbbs::uintE>& cluster_ids, ClusteringHelper* helper) {
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


  /*auto newnew = ;
  for (std::size_t i = 0; i < newnew->n; i++) {
    auto vtx = newnew->get_vertex(i);
    auto nbhrs = vtx.getOutNeighbors();
    double deg_i = vtx.getOutDegree();
    for (std::size_t j = 0; j < deg_i; j++) {
      new_node_weights[i] += std::get<1>(nbhrs[j]);
    }
  }*/

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

namespace {
// First, count triangles per edge (naive implementation)
void CountTriangles(
  gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
  gbbs::sequence<double>& triangle_counts, gbbs::sequence<gbbs::uintE>& degrees){
  // iterate through edges of the graph in order and do intersects
  pbbs::parallel_for(0, graph.n, [&](std::size_t i) {
    auto vtx = graph.get_vertex(i);
    //auto nbhrs = vtx.getOutNeighbors();
    //double deg_i = vtx.getOutDegree();
    for (std::size_t j = 0; j < vtx.getOutDegree(); j++) {
      auto j_idx = vtx.getOutNeighbor(j);
      auto vtx2 = graph.get_vertex(j_idx);
      // update triangle_counts[degrees[i] + j]
      triangle_counts[degrees[i] + j] = static_cast<double>(vtx.intersect(&vtx2, i, j_idx)) / 
        static_cast<double>(vtx.getOutDegree() + vtx2.getOutDegree());
    }
  });
}

// For each cluster, compute the subcluster ids
// return the next subcluster id
std::size_t ComputeSubclusterConnectivity(
  std::vector<ClusteringHelper::ClusterId>& subcluster_ids,
  std::size_t next_id, std::vector<gbbs::uintE>& cluster,
  gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph,
  std::vector<gbbs::uintE> all_cluster_ids,
  const ClustererConfig& clusterer_config) {
  // start everyone in a singleton neighborhood
  std::vector<gbbs::uintE> tmp_subclusters(cluster.size());
  pbbs::parallel_for(0, cluster.size(), [&] (size_t i) {
    tmp_subclusters[i] = i;
  });
  // 0 means it's a singleton
  std::vector<char> singletons(cluster.size(), 0);
  auto n = current_graph->n;
  auto get_clusters = [&](gbbs::uintE i) -> gbbs::uintE { return i; };
  std::vector<char> fast_intersect(n, 0);
  pbbs::parallel_for(0, cluster.size(), [&] (size_t i) {
    fast_intersect[cluster[i]] = 1;
  });
  for(std::size_t i = 0; i < cluster.size(); i++) {
    if (singletons[i] != 0) continue;
    auto curr_vtx = current_graph->get_vertex(cluster[i]);
    int curr_num_edges = 0;
    for (std::size_t j = 0; j < curr_vtx.getOutDegree(); j++) {
      if (fast_intersect[curr_vtx.getOutNeighbor(j)] == 0) curr_num_edges++;
    }
    if (curr_num_edges < cluster.size() - 1) continue;
    // Consider all clusters given by tmp_subclusters, that are
    // a) well-connected in relation to the set cluster
    // b) moving i to the cluster would net positive best moves value

    // First get the subclusters that satisfy the connectivity req
    // These are the current subclusters that i could move to
    // with values from 0 to cluster.size()
    auto curr_subclustering = 
      parallel::OutputIndicesById<ClusteringHelper::ClusterId, gbbs::uintE>(
        tmp_subclusters, get_clusters, cluster.size());
    // 0 means it's valid
    std::vector<char> valid_curr_subclustering(curr_subclustering.size(), 0);
    int num_valid = curr_subclustering.size();
    for (std::size_t j = 0; j < curr_subclustering.size(); j++) {
      // Check if the number of edges from curr_subclustering[j] to
      // cluster - curr_subclustering[j] is enough
      auto curr = curr_subclustering[j];
      if (curr.size() == 0) {
        valid_curr_subclustering[j] = 1;
        num_valid--;
        continue;
      }
      if (curr[0] == i) {
        valid_curr_subclustering[j] = 1;
        num_valid--;
        continue;
      }
      pbbs::parallel_for(0, curr.size(), [&] (size_t k) {
        fast_intersect[cluster[curr[k]]] = 2;
      });
      int num_edges = 0;
      for (std::size_t k = 0; k < curr.size(); k++) {
        auto vtx_idx = cluster[curr[k]];
        auto vtx = current_graph->get_vertex(vtx_idx);
        for (std::size_t l = 0; l < vtx.getOutDegree(); l++) {
          if (fast_intersect[vtx.getOutNeighbor(l)] == 0) num_edges++;
        }
      }
      pbbs::parallel_for(0, curr.size(), [&] (size_t k) {
        fast_intersect[cluster[curr[k]]] = 1;
      });
      // Invalid connectivity
      if (num_edges < curr.size() * (cluster.size() - curr.size())) {
        valid_curr_subclustering[j] = 1;
        num_valid--;
      }
    }
    if (num_valid == 0) continue;

    // TODO this is the very slow way
    //std::vector<gbbs::uintE> prev_cluster_ids = all_cluster_ids;
    //pbbs::parallel_for(0, cluster.size(), [&] (size_t k) {
    //  prev_cluster_ids[cluster[k]] = tmp_subclusters[k] + n;
    //});
    //InMemoryClusterer::Clustering prev_clustering = 
    //  parallel::OutputIndicesById<ClusteringHelper::ClusterId, int>(
    //    prev_cluster_ids, get_clusters, n);
    //double prev_objective = ComputeModularity(prev_clustering, *current_graph, 
    //  0, prev_cluster_ids);
    // For each valid curr_subclustering, check if it satisfies obj requirement
    double max_max = 0;
    std::size_t idx_idx = 0;
    for (std::size_t j = 0; j < curr_subclustering.size(); j++) {
      if (valid_curr_subclustering[j] != 0) continue;
      auto curr = curr_subclustering[j];
      pbbs::parallel_for(0, curr.size(), [&] (size_t k) {
        fast_intersect[cluster[curr[k]]] = 2;
      });
      double weight = 0;
      double offset = clusterer_config.correlation_clusterer_config().edge_weight_offset() * curr.size() + 
        clusterer_config.correlation_clusterer_config().resolution() * curr.size();
      for (std::size_t k = 0; k < curr_vtx.getOutDegree(); k++) {
        if (fast_intersect[curr_vtx.getOutNeighbor(k)] == 2)
          weight++;
      }
      pbbs::parallel_for(0, curr.size(), [&] (size_t k) {
        fast_intersect[cluster[curr[k]]] = 1;
      });
      if (weight < offset) {
        valid_curr_subclustering[j] = 1;
        num_valid--;
      } else if (weight - offset >= max_max) {
        max_max = weight-offset;
        idx_idx = j;
      }
      // Move i to the curr
      //prev_cluster_ids[cluster[i]] = n + tmp_subclusters[curr[0]];
      //InMemoryClusterer::Clustering next_clustering = 
      //  parallel::OutputIndicesById<ClusteringHelper::ClusterId, int>(
      //  prev_cluster_ids, get_clusters, n);
      //double next_objective = ComputeModularity(next_clustering, *current_graph, 
      //  0, prev_cluster_ids);
      //if (next_objective <= prev_objective) {
      //  valid_curr_subclustering[j] = 1;
      //  num_valid--;
      //}
    }
    if (num_valid == 0) continue;
    // There's at least one valid subclustering to move i to; do it
    // TODO this isn't randomly moving
    //for (std::size_t j = 0; j < curr_subclustering.size(); j++) {
      auto curr = curr_subclustering[idx_idx];
      //if (valid_curr_subclustering[j] == 0) {
        tmp_subclusters[i] = tmp_subclusters[curr[0]];
        singletons[i] = 1;
        singletons[curr[0]] = 1;
        //break;
      //}
    //}
  }
  std::size_t max_id = next_id;
  for(std::size_t i = 0; i < cluster.size(); i++) {
    subcluster_ids[cluster[i]] = tmp_subclusters[i] + next_id;
    if (tmp_subclusters[i] + next_id > max_id)
      max_id = tmp_subclusters[i] + next_id;
  }
  return max_id + 1;
}

std::size_t ComputeSubclusterTriangle(std::vector<ClusteringHelper::ClusterId>& subcluster_ids,
  std::size_t next_id, std::vector<gbbs::uintE>& cluster,
  gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph,
  CorrelationClustererSubclustering& subclustering,
  std::vector<gbbs::uintE> all_cluster_ids) {
  // The idea here is to form the subgraph on thresholded triangle counts
  // Maybe start with a threshold of 1?
  // Then, call connected components
  double boundary = 0.25;
  gbbs::sequence<gbbs::uintE> numbering(current_graph->n);
  pbbs::parallel_for(0, cluster.size(), [&] (size_t i) {
    numbering[cluster[i]] = i;
  });
  gbbs::sequence<gbbs::uintE> offsets(cluster.size() + 1);
  pbbs::parallel_for(0, cluster.size(), [&] (size_t i) {
    auto u = current_graph->get_vertex(cluster[i]);
    // For each j, the edge index is degrees[cluster[i]] + j
    offsets[i] = 0;
    auto map = [&](const gbbs::uintE u_id, const gbbs::uintE nbhr,
      float wgh, const gbbs::uintE j){
        bool pred = (subclustering.triangle_counts[subclustering.degrees[cluster[i]] + j] > boundary) &&
          all_cluster_ids[nbhr] == all_cluster_ids[cluster[i]];
        if (pred) offsets[i]++;
      };
    u.mapOutNghWithIndex(cluster[i], map);
  }, 1);
  offsets[cluster.size()] = 0;
  auto edge_count = pbbslib::scan_add_inplace(offsets);
  using edge = std::tuple<gbbs::uintE, pbbslib::empty>;
  auto edges = gbbs::sequence<edge>(edge_count);

  gbbs::parallel_for(0, cluster.size(), [&] (size_t i) {
    auto u = current_graph->get_vertex(cluster[i]);
    size_t out_offset = offsets[i];
    gbbs::uintE d = u.getOutDegree();
    auto nbhrs = u.getOutNeighbors();
    if (d > 0) {
      std::size_t idx = 0;
      for (std::size_t j = 0; j < d; j++) {
        if (subclustering.triangle_counts[subclustering.degrees[cluster[i]] + j] > boundary &&
          all_cluster_ids[std::get<0>(nbhrs[j])] == all_cluster_ids[cluster[i]]){
          edges[out_offset + idx] = std::make_tuple(numbering[u.getOutNeighbor(j)], pbbslib::empty());
          idx++;
        }
      }
    }
  }, 1);

  auto out_vdata = pbbs::new_array_no_init<gbbs::vertex_data>(cluster.size());
  gbbs::parallel_for(0, cluster.size(), [&] (size_t i) {
    out_vdata[i].offset = offsets[i];
    out_vdata[i].degree = offsets[i+1]-offsets[i];
  });
  offsets.clear();

  auto out_edge_arr = edges.to_array();
  auto G = gbbs::symmetric_graph<gbbs::symmetric_vertex, pbbslib::empty>(
      out_vdata, cluster.size(), edge_count,
      [=]() {pbbslib::free_arrays(out_vdata, out_edge_arr); },
      out_edge_arr);

  pbbs::sequence<gbbs::uintE> labels = gbbs::workefficient_cc::CC(G);
  auto max_label = pbbslib::reduce_max(labels);
  gbbs::parallel_for(0, cluster.size(), [&] (size_t i) {
    subcluster_ids[cluster[i]] = next_id + labels[i];
  });

  G.del();
  return max_label + next_id + 1;
}

}  // namespace internal

CorrelationClustererSubclustering::CorrelationClustererSubclustering(
  const ClustererConfig& clusterer_config,
  gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph){
  use_triangle = (clusterer_config.correlation_clusterer_config().subclustering_method() ==
    CorrelationClustererConfig::TRIANGLE_SUBCLUSTERING);
  if (use_triangle) {
    degrees = gbbs::sequence<gbbs::uintE>(current_graph->n, [&](std::size_t i) {
      return current_graph->get_vertex(i).getOutDegree();
    });
    auto total = pbbslib::scan_inplace(degrees.slice(), pbbslib::addm<gbbs::uintE>());
    triangle_counts = gbbs::sequence<double>(total, [](std::size_t i){ return 0; });
    CountTriangles(*current_graph, triangle_counts, degrees);
  }
}

std::size_t ComputeSubcluster(std::vector<ClusteringHelper::ClusterId>& subcluster_ids,
  std::size_t next_id, std::vector<gbbs::uintE>& cluster,
  gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph,
  CorrelationClustererSubclustering& subclustering,
  std::vector<gbbs::uintE> all_cluster_ids, const ClustererConfig& clusterer_config) {

  bool use_connectivity = (clusterer_config.correlation_clusterer_config()
    .subclustering_method() == CorrelationClustererConfig::CONNECTIVITY_SUBCLUSTERING);
  return use_connectivity ? 
    ComputeSubclusterConnectivity(subcluster_ids, next_id, cluster,
      current_graph, all_cluster_ids, clusterer_config) :
    ComputeSubclusterTriangle(subcluster_ids, next_id, cluster,
                      current_graph, subclustering,
                      all_cluster_ids);
}

}  // namespace in_memory
}  // namespace research_graph
