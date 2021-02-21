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
#include "external/gbbs/gbbs/pbbslib/sparse_table.h"

#include "external/gbbs/benchmarks/Connectivity/WorkEfficientSDB14/Connectivity.h"
#include "external/gbbs/benchmarks/Connectivity/SimpleUnionAsync/Connectivity.h"

namespace research_graph {
namespace in_memory {

float FloatFromWeightPCCI(float weight) { return weight; }
float FloatFromWeightPCCI(pbbslib::empty weight) { return 1; }

using NodeId = gbbs::uintE;
using ClusterId = gbbs::uintE;

void ClusteringHelper::ResetClustering(
  const std::vector<std::vector<gbbs::uintE>>& clustering) {
  pbbs::parallel_for(0, num_nodes_, [&](std::size_t i) {
      cluster_weights_[i] = 0;
      cluster_sizes_[i] = 0;
      
  });
  SetClustering(clustering);
}

void ClusteringHelper::SetClustering(
    const std::vector<std::vector<gbbs::uintE>>& clustering) {
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

/*
std::unique_ptr<bool[]> ClusteringHelper::MoveNodesToCluster(
  std::vector<absl::optional<ClusterId>>& moves,
  std::vector<double>& moves_obj,
  gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph) {
  const auto& config = clusterer_config_.correlation_clusterer_config();
  const double offset = config.edge_weight_offset();
  const double resolution = config.resolution();
  // Retrieve all nodes that are actually moving
  auto get_moving_nodes = [&](size_t i) { return i; };
  auto moving_nodes = pbbs::filter(
      pbbs::delayed_seq<gbbs::uintE>(num_nodes_, get_moving_nodes),
      [&](gbbs::uintE node) -> bool {
        return moves[node].has_value() && moves[node] != cluster_ids_[node];
      },
      pbbs::no_flag);
  if (moving_nodes.empty()) {
    auto modified_cluster = absl::make_unique<bool[]>(num_nodes_);
    pbbs::parallel_for(0, num_nodes_,
                     [&](std::size_t i) { modified_cluster[i] = false; });
    return modified_cluster;
  }
  // Sort moves by moves_obj
  using M = std::tuple<gbbs::uintE, double>;
  auto get_moves_func = [&](std::size_t j) {
    auto i = moving_nodes[j];
    return std::make_tuple(gbbs::uintE{i}, moves_obj[i]);
  };
  auto moves_sort = pbbs::sample_sort(
      pbbs::delayed_seq<M>(moving_nodes.size(), get_moves_func),
      [&](M a, M b) { return std::get<1>(a) > std::get<1>(b); }, true);
  
  //if (moves_sort.size() >= 1)
  //std::cout << "Max is first: " << std::get<1>(moves_sort[0]) << std::endl;
  // Make a rank array
  auto rank_array = gbbs::sequence<gbbs::uintE>(num_nodes_, [](std::size_t i){return 0;});
  pbbs::parallel_for(0, moves_sort.size(), [&](std::size_t i) {
    rank_array[std::get<0>(moves_sort[i])] = i;
  });
  // If your rank is higher, then you take the objective change
  pbbs::parallel_for(0, moves_sort.size(), [&](std::size_t i){
    auto vtx_idx = std::get<0>(moves_sort[i]);
    //if (moves[vtx_idx].has_value()) {
      auto d = moves[vtx_idx].value();
      auto c = cluster_ids_[vtx_idx];
      double obj_change = 0;
      auto map_moving_node_neighbors = [&](gbbs::uintE u, gbbs::uintE neighbor,
                                       double weight) {
      if (rank_array[u] > rank_array[neighbor] && moves[neighbor].has_value()) {
        auto b = moves[neighbor].value();
        auto a = cluster_ids_[neighbor];
        if (b == d) obj_change += weight - offset;
        if (a == d) obj_change -= weight - offset;
        if (b == c) obj_change -= weight - offset;
        if (a == c) obj_change += weight - offset;
      }
      };
      current_graph->get_vertex(vtx_idx)
        .mapOutNgh(vtx_idx, map_moving_node_neighbors, false);
      moves_sort[i] = std::make_tuple(vtx_idx, std::get<1>(moves_sort[i]) + obj_change);
    //}
  });
  // This is the inefficient n^2 thing
  pbbs::parallel_for(0, moves_sort.size(), [&](std::size_t i) {
    auto vtx_idx = std::get<0>(moves_sort[i]);
    auto d = moves[vtx_idx].value();
    auto c = cluster_ids_[vtx_idx];
    double obj_change = 0;
    for (std::size_t j = 0; j < i; j++) {
      auto u_id = std::get<0>(moves_sort[j]);
      auto b = moves[u_id].value();
      auto a = cluster_ids_[u_id];
      double change = -1 * resolution * node_weights_[vtx_idx] * node_weights_[u_id];
      if (b == d) obj_change += change;
      if (a == d) obj_change -= change;
      if (b == c) obj_change -= change;
      if (a == c) obj_change += change;
    }
    moves_sort[i] = std::make_tuple(vtx_idx, std::get<1>(moves_sort[i]) + obj_change);
  });
  // Now do a prefix sum on moves_sort
  auto f = [](const M& a, const M& b){
    return std::make_tuple(std::get<0>(b), std::get<1>(a) + std::get<1>(b));
  };
  auto prefix_sum_mon = pbbslib::make_monoid(f, std::make_tuple(gbbs::uintE{0}, double{0}));
  auto all = pbbs::scan_inplace(moves_sort.slice(), prefix_sum_mon);
  //if (moves_sort.size() >= 2)
  //std::cout << "Max is first2: " << std::get<0>(moves_sort[1]) << ", " << std::get<1>(moves_sort[1]) << std::endl;
  // Find the max move
  auto f_max = [](const M& a, const M& b){
    if (std::get<1>(a) > std::get<1>(b)) return a;
    return b;
  };
  auto max_monoid = pbbs::make_monoid(f_max, std::make_tuple(gbbs::uintE{0}, double{0}));
  auto max_move = pbbs::reduce(moves_sort.slice(), max_monoid);
  gbbs::uintE rank_max_move = rank_array[std::get<0>(all)];
  if (std::get<1>(max_move) > std::get<1>(all)) {
    rank_max_move = rank_array[std::get<0>(max_move)];
    //std::cout << "Positive Obj: " << std::get<1>(max_move) << std::endl;
  }
  //else {
    //std::cout << "Positive Obj: " << std::get<1>(all) << std::endl;
  //}
  pbbs::parallel_for(rank_max_move + 1, moves_sort.size(), [&](std::size_t i){
    moves[std::get<0>(moves_sort[i])] = absl::optional<ClusterId>();
  });
  return MoveNodesToCluster(moves);
}*/

/*
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
}*/

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



/*
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
}*/
/*
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
*/


}  // namespace in_memory
}  // namespace research_graph