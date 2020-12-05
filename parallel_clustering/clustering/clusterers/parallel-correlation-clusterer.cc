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

#include "clustering/clusterers/parallel-correlation-clusterer.h"

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "clustering/clusterers/parallel-correlation-clusterer-internal.h"
#include "clustering/config.pb.h"
#include "clustering/gbbs-graph.h"
#include "clustering/in-memory-clusterer.h"
#include "parallel/parallel-graph-utils.h"
#include "clustering/status_macros.h"

#include "external/gbbs/benchmarks/Connectivity/WorkEfficientSDB14/Connectivity.h"

namespace research_graph {
namespace in_memory {

namespace {

// This struct is necessary to perform an edge map with GBBS over a vertex
// set. Essentially, all neighbors are valid in this edge map, and this
// map does not do anything except allow for neighbors to be aggregated
// into the next frontier.
struct CorrelationClustererEdgeMap {
  inline bool cond(gbbs::uintE d) { return true; }
  inline bool update(const gbbs::uintE& s, const gbbs::uintE& d, float wgh) {
    return true;
  }
  inline bool updateAtomic(const gbbs::uintE& s, const gbbs::uintE& d,
                           float wgh) {
    return true;
  }
};

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

std::size_t ComputeSubcluster(std::vector<ClusteringHelper::ClusterId>& subcluster_ids,
  std::size_t next_id, std::vector<gbbs::uintE>& cluster,
  gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph,
  gbbs::sequence<double>&  triangle_counts, gbbs::sequence<gbbs::uintE>& degrees,
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
        bool pred = (triangle_counts[degrees[cluster[i]] + j] > boundary) &&
          all_cluster_ids[nbhr] == all_cluster_ids[cluster[i]];
        if (pred) offsets[i]++;
      };
    u.mapOutNghWithIndex(cluster[i], map);
/*
    auto out_f = [&](gbbs::uintE j) {
      bool pred = (triangle_counts[degrees[cluster[i]] + j] > boundary);
      return static_cast<int>(pred);
    };
    auto out_im = pbbslib::make_sequence<int>(u.getOutDegree(), out_f);
    if (out_im.size() > 0)
      offsets[i] = pbbslib::reduce_add(out_im);
    else
      offsets[i] = 0;*/
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
        if (triangle_counts[degrees[cluster[i]] + j] > boundary &&
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

// Given a vertex subset moved_subset, computes best moves for all vertices
// and performs the moves. Returns a vertex subset consisting of all vertices
// adjacent to modified clusters.
std::unique_ptr<gbbs::vertexSubset, void (*)(gbbs::vertexSubset*)>
BestMovesForVertexSubset(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph,
    std::size_t num_nodes, gbbs::vertexSubset* moved_subset,
    ClusteringHelper* helper, const ClustererConfig& clusterer_config,
    gbbs::sequence<double>& triangle_counts, gbbs::sequence<gbbs::uintE>& degrees) {
  std::vector<absl::optional<ClusteringHelper::ClusterId>> moves(num_nodes,
                                                                 absl::nullopt);

  // Find best moves per vertex in moved_subset
  //auto moved_clusters = absl::make_unique<bool[]>(current_graph->n);
  //pbbs::parallel_for(0, current_graph->n,
  //                   [&](std::size_t i) { moved_clusters[i] = false; });
  gbbs::vertexMap(*moved_subset, [&](std::size_t i) {
  //for (std::size_t i = 0; i < current_graph->n; i++) {
    //moved_clusters[i] = helper->AsyncMove(*current_graph, i);
    std::tuple<ClusteringHelper::ClusterId, double> best_move =
        helper->EfficientBestMove(*current_graph, i);
    // If a singleton cluster wishes to move to another singleton cluster,
    // only move if the id of the moving cluster is lower than the id
    // of the cluster it wishes to move to
    auto move_cluster_id = std::get<0>(best_move);
    auto current_cluster_id = helper->ClusterIds()[i];
    if (move_cluster_id < current_graph->n &&
        helper->ClusterSizes()[move_cluster_id] == 1 &&
        helper->ClusterSizes()[current_cluster_id] == 1 &&
        current_cluster_id >= move_cluster_id) {
      best_move = std::make_tuple(current_cluster_id, 0);
    }
    if (std::get<1>(best_move) > 0) moves[i] = std::get<0>(best_move);
  //}
  });

  // Compute modified clusters
  auto moved_clusters = helper->MoveNodesToCluster(moves);

  // Perform cluster moves
  if (clusterer_config.correlation_clusterer_config()
          .clustering_moves_method() ==
      CorrelationClustererConfig::DEFAULT_CLUSTER_MOVES) {
    // Reset moves
    pbbs::parallel_for(0, num_nodes,
                       [&](std::size_t i) { moves[i] = absl::nullopt; });

    // Aggregate clusters
    auto get_clusters = [&](gbbs::uintE i) -> gbbs::uintE { return i; };
    std::vector<std::vector<gbbs::uintE>> curr_clustering =
        parallel::OutputIndicesById<ClusteringHelper::ClusterId, gbbs::uintE>(
            helper->ClusterIds(), get_clusters, helper->ClusterIds().size());

    // ********* sub clusters
    // Compute sub-clusters by taking existing clusters and running
    // connected components on each cluster with triangle reweighting
    if (clusterer_config.correlation_clusterer_config()
          .subclustering_method() != CorrelationClustererConfig::NONE) {
      std::vector<ClusteringHelper::ClusterId> subcluster_ids(num_nodes);
      std::size_t next_id = 0;
      bool use_connectivity = (clusterer_config.correlation_clusterer_config()
          .subclustering_method() == CorrelationClustererConfig::CONNECTIVITY);
      for (std::size_t i = 0; i < curr_clustering.size(); i++) {
        next_id = use_connectivity ? 
        ComputeSubclusterConnectivity(subcluster_ids, next_id, curr_clustering[i],
          current_graph, helper->ClusterIds(), clusterer_config) :
        ComputeSubcluster(subcluster_ids, next_id, curr_clustering[i],
                           current_graph, triangle_counts, degrees,
                           helper->ClusterIds());
      }
      // now, do best cluster moves using subcluster ids
      std::vector<std::vector<gbbs::uintE>> curr_subclustering =
          parallel::OutputIndicesById<ClusteringHelper::ClusterId, gbbs::uintE>(
              subcluster_ids, get_clusters, num_nodes);
      // Compute best move per subcluster
      pbbs::parallel_for(0, curr_subclustering.size(), [&](std::size_t i) {
        if (!curr_subclustering[i].empty()) {
          std::tuple<ClusteringHelper::ClusterId, double> best_move =
              helper->BestMove(*current_graph, curr_subclustering[i]);
          // If a cluster wishes to move to another cluster,
          // only move if the id of the moving cluster is lower than the id
          // of the cluster it wishes to move to
          auto move_cluster_id = std::get<0>(best_move);
          auto current_cluster_id =
              helper->ClusterIds()[curr_subclustering[i].front()];
          if (move_cluster_id < current_graph->n &&
              current_cluster_id >= move_cluster_id) {
            best_move = std::make_tuple(current_cluster_id, 0);
          }
          if (std::get<1>(best_move) > 0) {
            for (size_t j = 0; j < curr_subclustering[i].size(); j++) {
              moves[curr_subclustering[i][j]] = std::get<0>(best_move);
            }
          }
        }
      });
      // Compute modified subclusters
      auto additional_moved_subclusters = helper->MoveNodesToCluster(moves);
      pbbs::parallel_for(0, num_nodes, [&](std::size_t i) {
        moved_clusters[i] |= additional_moved_subclusters[i];
      });
      // Reset moves
      pbbs::parallel_for(0, num_nodes,
                         [&](std::size_t i) { moves[i] = absl::nullopt; });
    }
    // ******** end sub clusters

    // Compute best move per cluster
    pbbs::parallel_for(0, curr_clustering.size(), [&](std::size_t i) {
      if (!curr_clustering[i].empty()) {
        std::tuple<ClusteringHelper::ClusterId, double> best_move =
            helper->BestMove(*current_graph, curr_clustering[i]);
        // If a cluster wishes to move to another cluster,
        // only move if the id of the moving cluster is lower than the id
        // of the cluster it wishes to move to
        auto move_cluster_id = std::get<0>(best_move);
        auto current_cluster_id =
            helper->ClusterIds()[curr_clustering[i].front()];
        if (move_cluster_id < current_graph->n &&
            current_cluster_id >= move_cluster_id) {
          best_move = std::make_tuple(current_cluster_id, 0);
        }
        if (std::get<1>(best_move) > 0) {
          for (size_t j = 0; j < curr_clustering[i].size(); j++) {
            moves[curr_clustering[i][j]] = std::get<0>(best_move);
          }
        }
      }
    });

    // Compute modified clusters
    auto additional_moved_clusters = helper->MoveNodesToCluster(moves);
    pbbs::parallel_for(0, num_nodes, [&](std::size_t i) {
      moved_clusters[i] |= additional_moved_clusters[i];
    });
  }

  // Mark vertices adjacent to clusters that have moved; these are
  // the vertices whose best moves must be recomputed.
  auto local_moved_subset =
      std::unique_ptr<gbbs::vertexSubset, void (*)(gbbs::vertexSubset*)>(
          new gbbs::vertexSubset(
              num_nodes, num_nodes,
              gbbs::sequence<bool>(
                  num_nodes,
                  [&](std::size_t i) {
                    return moved_clusters[helper->ClusterIds()[i]];
                  })
                  .to_array()),
          [](gbbs::vertexSubset* subset) {
            subset->del();
            delete subset;
          });
  auto edge_map = CorrelationClustererEdgeMap{};
  auto new_moved_subset =
      gbbs::edgeMap(*current_graph, *(local_moved_subset.get()), edge_map);
  return std::unique_ptr<gbbs::vertexSubset, void (*)(gbbs::vertexSubset*)>(
      new gbbs::vertexSubset(std::move(new_moved_subset)),
      [](gbbs::vertexSubset* subset) {
        subset->del();
        delete subset;
      });
}

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

}  // namespace

absl::Status ParallelCorrelationClusterer::RefineClusters(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering) const {
  std::vector<double> empty;
  return RefineClusters(clusterer_config, initial_clustering, empty);
}

absl::Status ParallelCorrelationClusterer::RefineClusters(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    std::vector<double> node_weights) const {
  const auto& config = clusterer_config.correlation_clusterer_config();

  std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>
      compressed_graph;

  // Set number of iterations based on clustering method
  int num_iterations = 0;
  int num_inner_iterations = 0;
  switch (config.clustering_moves_method()) {
    case CorrelationClustererConfig::DEFAULT_CLUSTER_MOVES:
      num_iterations = 1;
      num_inner_iterations =
          config.num_iterations() > 0 ? config.num_iterations() : 32;
      break;
    case CorrelationClustererConfig::LOUVAIN:
      num_iterations = config.louvain_config().num_iterations() > 0
                           ? config.louvain_config().num_iterations()
                           : 32;
      num_inner_iterations =
          config.louvain_config().num_inner_iterations() > 0
              ? config.louvain_config().num_inner_iterations()
              : 32;
      break;
    default:
      return absl::UnimplementedError(
          "Correlation clustering moves must be DEFAULT_CLUSTER_MOVES or "
          "LOUVAIN.");
  }

  // Initialize clustering helper
  auto helper = node_weights.empty() ? absl::make_unique<ClusteringHelper>(
      graph_.Graph()->n, clusterer_config, *initial_clustering) :
      absl::make_unique<ClusteringHelper>(
      graph_.Graph()->n, clusterer_config, node_weights, *initial_clustering);
  // The max objective is the maximum objective given by the inner iterations
  // of best moves rounds
  double max_objective = helper->ComputeObjective(*(graph_.Graph()));

  std::vector<gbbs::uintE> cluster_ids(graph_.Graph()->n);
  std::vector<gbbs::uintE> local_cluster_ids(graph_.Graph()->n);
  pbbs::parallel_for(0, graph_.Graph()->n, [&](std::size_t i) {
    cluster_ids[i] = i;
    local_cluster_ids[i] = helper->ClusterIds()[i];
  });
  //int iter = 0;
  //while (true) {
  for (int iter = 0; iter < num_iterations; ++iter) {
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph =
        (iter == 0) ? graph_.Graph() : compressed_graph.get();
    const auto num_nodes = current_graph->n;

    // *****triangle counting
    // TODO this is kinda hackish
    bool use_triangle = (config.subclustering_method() == CorrelationClustererConfig::TRIANGLE);
    gbbs::sequence<gbbs::uintE> degrees(use_triangle ? num_nodes : 0,
                  [&](std::size_t i) {
                    return current_graph->get_vertex(i).getOutDegree();
                  });
    auto total = use_triangle ? pbbslib::scan_inplace(degrees.slice(), pbbslib::addm<gbbs::uintE>()) : 0;
    gbbs::sequence<double> triangle_counts(use_triangle ? total : 0, [](std::size_t i){ return 0; });
    if (use_triangle) CountTriangles(*current_graph, triangle_counts, degrees);

    bool moved = false;
    bool local_moved = true;
    auto moved_subset =
        std::unique_ptr<gbbs::vertexSubset, void (*)(gbbs::vertexSubset*)>(
            new gbbs::vertexSubset(
                num_nodes, num_nodes,
                gbbs::sequence<bool>(num_nodes, true).to_array()),
            [](gbbs::vertexSubset* subset) {
              subset->del();
              delete subset;
            });

    // Iterate over best moves
    for (int local_iter = 0; local_iter < num_inner_iterations && local_moved;
         ++local_iter) {
    //double prev_objective = 0;
    //while (local_moved) {
      auto new_moved_subset =
          BestMovesForVertexSubset(current_graph, num_nodes, moved_subset.get(),
                                   helper.get(), clusterer_config, triangle_counts,
                                   degrees);
      moved_subset.swap(new_moved_subset);
      local_moved = !moved_subset->isEmpty();

      // Compute new objective given by the local moves in this iteration
      double curr_objective = helper->ComputeObjective(*current_graph);

      // Update maximum objective
      //if (curr_objective > max_objective) {
        //pbbs::parallel_for(0, num_nodes, [&](std::size_t i) {
        //  local_cluster_ids[i] = helper->ClusterIds()[i];
        //});
        max_objective = curr_objective;
        moved |= local_moved;
      //}
      //if (prev_objective == curr_objective || abs(curr_objective - prev_objective) / current_graph->m < 0.0001)
      //  break;
      //prev_objective = curr_objective;
std::cout << "Curr: " << curr_objective << std::endl;
std::cout << "Max: " << max_objective << std::endl;
    }

    pbbs::parallel_for(0, num_nodes, [&](std::size_t i) {
      local_cluster_ids[i] = helper->ClusterIds()[i];
    });

    // If no moves can be made at all, exit
    if (!moved) break;

    // Compress cluster ids in initial_helper based on helper
    cluster_ids = FlattenClustering(cluster_ids, local_cluster_ids);

    if (iter == num_iterations - 1) break;

    if (config.subclustering_method() != CorrelationClustererConfig::NONE) {
      // ***** subclusters from cluster ids
      auto get_clusters = [&](gbbs::uintE i) -> gbbs::uintE { return i; };
      std::vector<std::vector<gbbs::uintE>> curr_clustering =
          parallel::OutputIndicesById<ClusteringHelper::ClusterId, gbbs::uintE>(
              local_cluster_ids, get_clusters, num_nodes);
      std::vector<ClusteringHelper::ClusterId> subcluster_ids(num_nodes);
      std::vector<std::size_t> next_ids(curr_clustering.size() + 1);
      std::size_t next_id = 0;
      next_ids[0] = next_id;
      bool use_connectivity = 
        (config.subclustering_method() == CorrelationClustererConfig::CONNECTIVITY);
      for (std::size_t i = 0; i < curr_clustering.size(); i++) {
        // When we create new local clusters, both with local_cluster_ids and helper,
        // it should map the vertices given by prev_id to next_id, to
        // local_cluster_ids[curr_clustering[i][0]] 
        next_id = use_connectivity ? 
        ComputeSubclusterConnectivity(subcluster_ids, next_id, curr_clustering[i],
          current_graph, local_cluster_ids, clusterer_config) :
        ComputeSubcluster(subcluster_ids, next_id, curr_clustering[i],
                           current_graph, triangle_counts, degrees,
                           local_cluster_ids);
        next_ids[i + 1] = next_id;
      }

      // Create new local clusters (subcluster)
      //std::vector<gbbs::uintE> local_cluster_ids2 = local_cluster_ids;
      pbbs::parallel_for(1, curr_clustering.size() + 1, [&](std::size_t i) {
        for (std::size_t j = next_ids[i-1]; j < next_ids[i]; j++) {
          local_cluster_ids[j] = i-1;//local_cluster_ids2[curr_clustering[i][0]];
        }
      });
      InMemoryClusterer::Clustering new_clustering =
          parallel::OutputIndicesById<ClusteringHelper::ClusterId, int>(
              local_cluster_ids, get_clusters, next_id);

      // TODO(jeshi): May want to compress out size 0 clusters when compressing
      // the graph
      GraphWithWeights new_compressed_graph;
      // **** replace cluster id compression with subcluster
      ASSIGN_OR_RETURN(
          new_compressed_graph,
          CompressGraph(*current_graph, subcluster_ids, helper.get()));
      //ASSIGN_OR_RETURN(
      //    new_compressed_graph,
      //    CompressGraph(*current_graph, local_cluster_ids, helper.get()));
      compressed_graph.swap(new_compressed_graph.graph);
      if (new_compressed_graph.graph) new_compressed_graph.graph->del();

      helper = absl::make_unique<ClusteringHelper>(
          compressed_graph->n, clusterer_config,
          new_compressed_graph.node_weights, new_clustering);
    } else if (config.subclustering_method() == CorrelationClustererConfig::NONE) {
      // TODO(jeshi): May want to compress out size 0 clusters when compressing
      // the graph
      GraphWithWeights new_compressed_graph;
      ASSIGN_OR_RETURN(
          new_compressed_graph,
          CompressGraph(*current_graph, local_cluster_ids, helper.get()));
      compressed_graph.swap(new_compressed_graph.graph);
      if (new_compressed_graph.graph) new_compressed_graph.graph->del();
      helper = absl::make_unique<ClusteringHelper>(
          compressed_graph->n, clusterer_config,
          new_compressed_graph.node_weights, InMemoryClusterer::Clustering{});

      // Create new local clusters
      pbbs::parallel_for(0, compressed_graph->n,
                         [&](std::size_t i) { local_cluster_ids[i] = i; });
    }
    //iter++;
  }

  if (compressed_graph) compressed_graph->del();

  auto get_clusters = [&](NodeId i) -> NodeId { return i; };

  *initial_clustering = parallel::OutputIndicesById<ClusterId, NodeId>(
      cluster_ids, get_clusters, cluster_ids.size());

  return absl::OkStatus();
}

absl::StatusOr<InMemoryClusterer::Clustering>
ParallelCorrelationClusterer::Cluster(
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
