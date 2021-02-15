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

#include "external/gbbs/benchmarks/KCore/JulienneDBS17/KCore.h"
#include "external/gbbs/gbbs/pbbslib/sparse_additive_map.h"
#include "external/gbbs/pbbslib/random_shuffle.h"
//#include "external/gbbs/pbbslib/union_find.h"

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

struct CorrelationClustererRefine {
  using H = std::unique_ptr<ClusteringHelper>;
  using G = std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>;
  gbbs::sequence<H> recurse_helpers = gbbs::sequence<H>(0, [](std::size_t i){return H(nullptr);});
  gbbs::sequence<G> recurse_graphs = gbbs::sequence<G>(0, [](std::size_t i){return G(nullptr);});
  bool use_refine = false;
};

// Given a vertex subset moved_subset, computes best moves for all vertices
// and performs the moves. Returns a vertex subset consisting of all vertices
// adjacent to modified clusters.
std::unique_ptr<gbbs::vertexSubset, void (*)(gbbs::vertexSubset*)>
BestMovesForVertexSubset(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph,
    std::size_t num_nodes, gbbs::vertexSubset* moved_subset,
    ClusteringHelper* helper, const ClustererConfig& clusterer_config,
    CorrelationClustererSubclustering& subclustering) {
  bool async = clusterer_config.correlation_clusterer_config().async();
  std::vector<absl::optional<ClusteringHelper::ClusterId>> moves(num_nodes,
                                                                 absl::nullopt);
  std::vector<double> moves_obj(num_nodes, 0);

  //pbbs::sequence<char> moves_bool(moved_subset->size(), [&](std::size_t i){
  //  if (i > ((double) moved_subset->size()) / 2.0) return 1;
  //  return 0;
  //});
  //auto moves_bool_shuffle = pbbs::random_shuffle(moves_bool.slice());

  // Find best moves per vertex in moved_subset
  auto moved_clusters = absl::make_unique<bool[]>(current_graph->n);
  pbbs::parallel_for(0, current_graph->n,
                     [&](std::size_t i) { moved_clusters[i] = false; });
  gbbs::vertexMap(*moved_subset, [&](std::size_t i) {
  //for (std::size_t i = 0; i < current_graph->n; i++) {
    if (async) {
      moved_clusters[i] = helper->AsyncMove(*current_graph, i);
    } else {//if (moves_bool_shuffle[i] == 1){
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
    if (std::get<1>(best_move) > 0) {
      moves[i] = std::get<0>(best_move);
      moves_obj[i] = std::get<1>(best_move);
    }
    }
  });

  // Compute modified clusters
  if (!async) {
    moved_clusters = helper->MoveNodesToCluster(moves);//, moves_obj, current_graph);
    /*pbbs::parallel_for(0, num_nodes,
                       [&](std::size_t i) { moves[i] = absl::nullopt; });
    gbbs::vertexMap(*moved_subset, [&](std::size_t i) {
  //for (std::size_t i = 0; i < current_graph->n; i++) {
      if (moves_bool_shuffle[i] != 1){
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
    if (std::get<1>(best_move) > 0) {
      moves[i] = std::get<0>(best_move);
      moves_obj[i] = std::get<1>(best_move);
    }
      }
    });
    auto more_moved_clusters = helper->MoveNodesToCluster(moves);//, moves_obj, current_graph);
    pbbs::parallel_for(0, num_nodes, [&](std::size_t i) {
        moved_clusters[i] |= more_moved_clusters[i];
      });*/
  }

  // Perform cluster moves
  if (clusterer_config.correlation_clusterer_config()
          .clustering_moves_method() ==
      CorrelationClustererConfig::DEFAULT_CLUSTER_MOVES) {
    // Reset moves
    if (!async) {
      pbbs::parallel_for(0, num_nodes,
                       [&](std::size_t i) { moves[i] = absl::nullopt; });
    }

    // Aggregate clusters
    auto get_clusters = [&](gbbs::uintE i) -> gbbs::uintE { return i; };
    std::vector<std::vector<gbbs::uintE>> curr_clustering =
        parallel::OutputIndicesById<ClusteringHelper::ClusterId, gbbs::uintE>(
            helper->ClusterIds(), get_clusters, helper->ClusterIds().size());

    // ********* sub clusters
    // Compute sub-clusters by taking existing clusters and running
    // connected components on each cluster with triangle reweighting
    if (clusterer_config.correlation_clusterer_config()
          .subclustering_method() != CorrelationClustererConfig::NONE_SUBCLUSTERING) {
      std::vector<ClusteringHelper::ClusterId> subcluster_ids(num_nodes);
      std::size_t next_id = 0;
      for (std::size_t i = 0; i < curr_clustering.size(); i++) {
        next_id = ComputeSubcluster(subcluster_ids, next_id, curr_clustering[i], current_graph,
          subclustering, helper->ClusterIds(), clusterer_config);
      }
      // now, do best cluster moves using subcluster ids
      std::vector<std::vector<gbbs::uintE>> curr_subclustering =
          parallel::OutputIndicesById<ClusteringHelper::ClusterId, gbbs::uintE>(
              subcluster_ids, get_clusters, num_nodes);
      // Compute best move per subcluster
      auto additional_moved_subclusters = absl::make_unique<bool[]>(current_graph->n);
      if (async) {
      pbbs::parallel_for(0, current_graph->n,
                     [&](std::size_t i) { additional_moved_subclusters[i] = false; });
      }
      pbbs::parallel_for(0, curr_subclustering.size(), [&](std::size_t i) {
        if (!curr_subclustering[i].empty()) {
          if (async) {
            bool move_flag = helper->AsyncMove(*current_graph, curr_subclustering[i]);
            if (move_flag) {
              pbbs::parallel_for(0, curr_subclustering[i].size(), [&](std::size_t j) {
                additional_moved_subclusters[curr_clustering[i][j]] = true;
              });
            }
          }
          else {
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
        }
      });
      // Compute modified subclusters
      if (!async) additional_moved_subclusters = helper->MoveNodesToCluster(moves);
      pbbs::parallel_for(0, num_nodes, [&](std::size_t i) {
        moved_clusters[i] |= additional_moved_subclusters[i];
      });
      // Reset moves
      if (!async) {
        pbbs::parallel_for(0, num_nodes,
                         [&](std::size_t i) { moves[i] = absl::nullopt; });
      }
    }
    // ******** end sub clusters

    // Compute best move per cluster
    auto additional_moved_clusters = absl::make_unique<bool[]>(current_graph->n);
    if (async) {
    pbbs::parallel_for(0, current_graph->n,
                     [&](std::size_t i) { additional_moved_clusters[i] = false; });
    }
    pbbs::parallel_for(0, curr_clustering.size(), [&](std::size_t i) {
      if (!curr_clustering[i].empty()) {
        if (async) {
          bool move_flag = helper->AsyncMove(*current_graph, curr_clustering[i]);
          if (move_flag) {
            pbbs::parallel_for(0, curr_clustering[i].size(), [&](std::size_t j) {
              additional_moved_clusters[curr_clustering[i][j]] = true;
            });
          }
        } else {
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
      }
    });

    // Compute modified clusters
    if(!async) additional_moved_clusters = helper->MoveNodesToCluster(moves);
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

bool IterateBestMoves(int num_inner_iterations, const ClustererConfig& clusterer_config,
  gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph,
  ClusteringHelper* helper, CorrelationClustererSubclustering& subclustering) {
  const auto num_nodes = current_graph->n;
  bool moved = false;
  bool local_moved = true;
  auto moved_subset = std::unique_ptr<gbbs::vertexSubset, void (*)(gbbs::vertexSubset*)>(
    new gbbs::vertexSubset(num_nodes, num_nodes,
    gbbs::sequence<bool>(num_nodes, true).to_array()),
    [](gbbs::vertexSubset* subset) {
      subset->del();
      delete subset;
    });

  // Iterate over best moves
  int local_iter = 0;
  for (local_iter = 0; local_iter < num_inner_iterations && local_moved; ++local_iter) {
    auto new_moved_subset =
      BestMovesForVertexSubset(current_graph, num_nodes, moved_subset.get(),
                              helper, clusterer_config, subclustering);
    moved_subset.swap(new_moved_subset);
    local_moved = !moved_subset->isEmpty();
    moved |= local_moved;
  }
  std::cout << "Num inner: " << local_iter + 1 << std::endl;
  return moved;
}

}  // namespace

template <template <class inner_wgh> class vtx_type, class wgh_type,
          typename P,
          typename std::enable_if<
              std::is_same<vtx_type<wgh_type>, gbbs::symmetric_vertex<wgh_type>>::value,
              int>::type = 0>
static inline gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, wgh_type> filterGraph(
    gbbs::symmetric_ptr_graph<vtx_type, wgh_type>& G, P& pred) {
  auto[newN, newM, newVData, newEdges] = gbbs::filter_graph<vtx_type, wgh_type>(G, pred);
  assert(newN == G.num_vertices());
  auto out_vdata = pbbs::new_array_no_init<gbbs::symmetric_vertex<float>>(newN);
  pbbs::parallel_for(0, newN, [&] (size_t i) {
    auto offset = (i == newN - 1) ? newM : newVData[i+1].offset;
    out_vdata[i].degree = offset-newVData[i].offset;
    out_vdata[i].neighbors = newEdges + newVData[i].offset;
  });
  pbbslib::free_arrays(newVData);
  return gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, wgh_type>(
      newN, newM, out_vdata,
      [newVData = out_vdata, newEdges = newEdges]() {
        pbbslib::free_arrays(newVData, newEdges);
      });
}

absl::Status ParallelCorrelationClusterer::RefineClusters(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering) const {
  std::vector<double> empty;
  return RefineClusters(clusterer_config, initial_clustering, empty);
}

absl::Status ParallelCorrelationClusterer::RefineClusters(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* graph) const {
  std::vector<double> empty;
  return RefineClusters(clusterer_config, initial_clustering, empty, graph);
}

absl::Status ParallelCorrelationClusterer::RefineClusters(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    std::vector<double> node_weights, double resolution) const {
    return RefineClusters(clusterer_config, initial_clustering, node_weights, graph_.Graph(), resolution);
}

absl::Status ParallelCorrelationClusterer::RefineClusters_subroutine(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering) const {
  std::vector<double> empty;
  return RefineClusters_subroutine(clusterer_config, initial_clustering, empty);
}

absl::Status ParallelCorrelationClusterer::RefineClusters_subroutine(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* graph) const {
  std::vector<double> empty;
  return RefineClusters_subroutine(clusterer_config, initial_clustering, empty, graph);
}

absl::Status ParallelCorrelationClusterer::RefineClusters_subroutine(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    std::vector<double> node_weights) const {
    return RefineClusters_subroutine(clusterer_config, initial_clustering, node_weights, graph_.Graph());
}

absl::StatusOr<GraphWithWeights> CompressSubclusters(const ClustererConfig& clusterer_config, 
  gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph,
  std::vector<gbbs::uintE>& local_cluster_ids, 
  ClusteringHelper* helper,
  CorrelationClustererSubclustering& subclustering,
  InMemoryClusterer::Clustering& new_clustering) {
  // ***** subclusters from cluster ids
  auto get_clusters = [&](gbbs::uintE i) -> gbbs::uintE { return i; };
  std::vector<std::vector<gbbs::uintE>> curr_clustering =
    parallel::OutputIndicesById<ClusteringHelper::ClusterId, gbbs::uintE>(
      local_cluster_ids, get_clusters, current_graph->n);
  std::vector<ClusteringHelper::ClusterId> subcluster_ids(current_graph->n);
  std::vector<std::size_t> next_ids(curr_clustering.size() + 1);
  std::size_t next_id = 0;
  next_ids[0] = next_id;
  for (std::size_t i = 0; i < curr_clustering.size(); i++) {
    // When we create new local clusters, both with local_cluster_ids and helper,
    // it should map the vertices given by prev_id to next_id, to
    // local_cluster_ids[curr_clustering[i][0]] 
    next_id = ComputeSubcluster(subcluster_ids, next_id, curr_clustering[i], current_graph,
      subclustering, local_cluster_ids, clusterer_config);
    next_ids[i + 1] = next_id;
  }

  // Create new local clusters (subcluster)
  pbbs::parallel_for(1, curr_clustering.size() + 1, [&](std::size_t i) {
    for (std::size_t j = next_ids[i-1]; j < next_ids[i]; j++) {
      local_cluster_ids[j] = i-1;
    }
  });
  new_clustering = parallel::OutputIndicesById<ClusteringHelper::ClusterId, int>(
    local_cluster_ids, get_clusters, next_id);

  return CompressGraph(*current_graph, subcluster_ids, helper);
} 

absl::Status ParallelCorrelationClusterer::RefineClusters_subroutine(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    std::vector<double> node_weights,
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* graph) const {
      std::cout << "REFINE" << std::endl;
      fflush(stdout);
    const auto& config = clusterer_config.correlation_clusterer_config();
  // Set number of iterations based on clustering method
  int num_iterations = 0;
  int num_inner_iterations = 0;
  switch (config.clustering_moves_method()) {
    case CorrelationClustererConfig::DEFAULT_CLUSTER_MOVES:
      num_iterations = 1;
      num_inner_iterations =
          config.num_iterations() > 0 ? config.num_iterations() : 64;
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

  // Initialize refinement data structure
  CorrelationClustererRefine refine{};
  if (config.refine()) {
    using H = std::unique_ptr<ClusteringHelper>;
    using G = std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>;
    refine.recurse_helpers = gbbs::sequence<H>(num_iterations, [](std::size_t i){return H(nullptr);});
    refine.recurse_graphs = gbbs::sequence<G>(num_iterations, [](std::size_t i){return G(nullptr);});
  }

  std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>
      compressed_graph;

  // Initialize clustering helper
  auto helper = node_weights.empty() ? absl::make_unique<ClusteringHelper>(
      graph->n, clusterer_config, *initial_clustering) :
      absl::make_unique<ClusteringHelper>(
      graph->n, clusterer_config, node_weights, *initial_clustering);
  // The max objective is the maximum objective given by the inner iterations
  // of best moves rounds
  //double max_objective = helper->ComputeObjective(*graph);

  std::vector<gbbs::uintE> cluster_ids(graph->n);
  std::vector<gbbs::uintE> local_cluster_ids(graph->n);
  pbbs::parallel_for(0, graph->n, [&](std::size_t i) {
    cluster_ids[i] = i;
  });

  int iter = 0;
  for (iter = 0; iter < num_iterations; ++iter) {
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph =
        (iter == 0) ? graph : compressed_graph.get();
    // Initialize subclustering data structure
    CorrelationClustererSubclustering subclustering(clusterer_config, current_graph);
    bool moved = IterateBestMoves(num_inner_iterations, clusterer_config, current_graph,
      helper.get(), subclustering);

    // If no moves can be made at all, exit
    if (!moved) {
      iter--;
      break;
    }

    // Compress cluster ids in initial_helper based on helper
    if (!config.refine()) cluster_ids = FlattenClustering(cluster_ids, helper->ClusterIds());
    else if (config.refine() && iter == num_iterations - 1) {
      refine.recurse_helpers[iter] = std::move(helper);
      refine.recurse_graphs[iter] = (iter == 0) ? nullptr : std::move(compressed_graph);
    } 

    if (iter == num_iterations - 1) break;

    // Compress graph
    GraphWithWeights new_compressed_graph;
    InMemoryClusterer::Clustering new_clustering{};
    pbbs::parallel_for(0, current_graph->n, [&](std::size_t i) {
        local_cluster_ids[i] = helper->ClusterIds()[i];
    });
    if (config.subclustering_method() != CorrelationClustererConfig::NONE_SUBCLUSTERING) {
      ASSIGN_OR_RETURN(new_compressed_graph,
        CompressSubclusters(clusterer_config, current_graph, local_cluster_ids, helper.get(),
        subclustering, new_clustering));
    } else {
      ASSIGN_OR_RETURN(
          new_compressed_graph,
          CompressGraph(*current_graph, local_cluster_ids, helper.get()));
      // Create new local clusters
      pbbs::parallel_for(0, new_compressed_graph.graph->n,
                         [&](std::size_t i) { local_cluster_ids[i] = i; });
    }

    compressed_graph.swap(new_compressed_graph.graph);
    if (config.refine()) {
      refine.recurse_helpers[iter] = std::move(helper);
      refine.recurse_graphs[iter] = std::move(new_compressed_graph.graph);
    } else if (new_compressed_graph.graph) new_compressed_graph.graph->del();

    helper = absl::make_unique<ClusteringHelper>(
        compressed_graph->n, clusterer_config,
        new_compressed_graph.node_weights, new_clustering);
  }

  // Refine clusters up the stack
  if (config.refine()) {
    auto get_clusters = [&](NodeId i) -> NodeId { return i; };
    for (int i = iter - 1; i >= 0; i--) {
      gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph =
        (i == 0) ? graph : refine.recurse_graphs[i].get();
      //assert (current_graph != nullptr);
      //if (current_graph == nullptr) std::cout << "null" << std::endl;
      //fflush(stdout);
      //assert (refine.recurse_helpers[i+1].get() != nullptr);
      //if (refine.recurse_helpers[i+1].get() == nullptr) {
      //  std::cout << "null help" << std::endl;
      //  fflush(stdout);
     // }
      //std::cout << "REFINE: " << i << std::endl;
      auto flatten_cluster_ids = FlattenClustering(refine.recurse_helpers[i]->ClusterIds(),
          refine.recurse_helpers[i+1]->ClusterIds());
      auto flatten_clustering = parallel::OutputIndicesById<ClusterId, NodeId>(
          flatten_cluster_ids,
          get_clusters, 
          flatten_cluster_ids.size());
      refine.recurse_helpers[i]->ResetClustering(flatten_clustering);
      //std::cout << "REFINE2: " << i << std::endl;
      // Take iter + 1 helper; use it to compress into iter helper; then
      // do best moves again, using the graph and node weights from iter

      // TODO: get rid of subclustering here
      CorrelationClustererSubclustering subclustering(clusterer_config, current_graph);
      IterateBestMoves(num_inner_iterations, clusterer_config, current_graph,
        refine.recurse_helpers[i].get(), subclustering);
    }
    cluster_ids = refine.recurse_helpers[0]->ClusterIds();
  }

  std::cout << "Num outer: " << iter + 1 << std::endl;

  if (compressed_graph) compressed_graph->del();

  auto get_clusters = [&](NodeId i) -> NodeId { return i; };

  *initial_clustering = parallel::OutputIndicesById<ClusterId, NodeId>(
      cluster_ids, get_clusters, cluster_ids.size());

  return absl::OkStatus();
}

/*
// The following supports both "union" that is only safe sequentially
// and "link" that is safe in parallel.  Find is always safe in parallel.
// See:  "Internally deterministic parallel algorithms can be fast"
// Blelloch, Fineman, Gibbons, and Shun
// for a discussion of link/find.
template <class vertexId>
struct unionFind {
  pbbs::sequence<vertexId> parents;

  bool is_root(vertexId u) { return parents[u] < 0; }

  // initialize n elements all as roots
  unionFind(size_t n) { parents = pbbs::sequence<vertexId>(n, [](std::size_t i){return -1;}); }

  vertexId find(vertexId i) {
    if (is_root(i)) return i;
    vertexId p = parents[i];
    if (is_root(p)) return p;

    // find root, shortcutting along the way
    do {
      vertexId gp = parents[p];
      parents[i] = gp;
      i = p;
      p = gp;
    } while (!is_root(p));
    return p;
  }

  // If using "union" then "parents" are used both as
  // parent pointer and for rank (a bit of a hack).
  // When a vertex is a root (negative) then the magnitude
  // of the negative number is its rank.
  // Otherwise it is the vertexId of its parent.
  // cannot be called union since reserved in C
  void union_roots(vertexId u, vertexId v) {
    if (parents[v] < parents[u]) std::swap(u, v);
    // now u has higher rank (higher negative number)
    parents[u] += parents[v];  // update rank of root
    parents[v] = u;            // update parent of other tree
  }

  // Version of union that is safe for parallelism
  // when no cycles are created (e.g. only link from larger
  // to smaller vertexId).
  // Does not use ranks.
  void link(vertexId u, vertexId v) { parents[u] = v; }

  // returns true if successful
  bool tryLink(vertexId u, vertexId v) {
    return (parents[u] == -1 &&
            pbbs::atomic_compare_and_swap(&parents[u], -1, v));
  }
};


template<class G, class parent, class F>
void compute_components2(std::vector<parent>& parents, G& GA, F func) {
    using W = typename G::weight_type;
    size_t n = GA.n;
    //gbbs::uintE granularity = 1;
    unionFind<int> union_find(n);

    pbbs::parallel_for(0, n, [&] (std::size_t i) {
      auto map_f = [&] (gbbs::uintE u, gbbs::uintE v, const W& wgh) {
        if (v < u && func(v)) {
          union_find.link(u, v);
        }
      };
      if (func(i)) GA.get_vertex(i).mapOutNgh(i, map_f);
    });

    pbbs::parallel_for(0, n, [&] (std::size_t i) {
      parents[i] = union_find.find(i);
    });
  }
*/
absl::Status ParallelCorrelationClusterer::RefineClusters(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    std::vector<double> node_weights,
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* graph, double original_resolution) const {
  const auto& config = clusterer_config.correlation_clusterer_config();
  if (config.preclustering_method() == CorrelationClustererConfig::KCORE_PRECLUSTERING) {
      // First, run a truncated k-core
      // TODO: this is not truncated

auto begin = std::chrono::steady_clock::now();
      auto cores = gbbs::KCore(*(graph_.Graph()), 16);
auto end = std::chrono::steady_clock::now();
std::cout << "KCore Time: " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  <<std::endl;
      
/*if (config.kcore_config().connect_only()) {
      std::vector<gbbs::uintE> ids((graph_.Graph())->n);
      auto f = [&](gbbs::uintE u) -> bool {
        return cores[u] >= config.kcore_config().kcore_cutoff();
      };
      compute_components2(ids, *(graph_.Graph()), f);
      auto get_clusters = [&](NodeId i) -> NodeId { return i; };
      *initial_clustering = parallel::OutputIndicesById<ClusterId, NodeId>(
        ids, get_clusters, ids.size());
} else {*/

auto cutoff = config.kcore_config().kcore_cutoff();
if (cutoff != 0) {
      auto num_in_cores = pbbslib::make_sequence<gbbs::uintE>(graph_.Graph()->n, [&](std::size_t i){
        if (cores[i] >= cutoff) return gbbs::uintE{1};
        return gbbs::uintE{0};
      });
      auto num_in_core = pbbslib::reduce_add(num_in_cores);
      double percent_in_core = num_in_core / (double) graph_.Graph()->n;
      std::cout << "Num in core: " << num_in_core << std::endl;
      std::cout << "Percent in core: " << percent_in_core << std::endl;
      gbbs::uintE num_edges_in_core = 0;
      for (std::size_t i = 0; i < graph_.Graph()->n; i++) {
        if (cores[i] < cutoff) continue;
        auto vtx = graph_.Graph()->get_vertex(i);
        for (std::size_t j = 0; j < vtx.getOutDegree(); j++) {
          if (cores[vtx.getOutNeighbor(j)] >= cutoff) num_edges_in_core++;
        }
      }
      std::cout << "Num edges in core: " << num_edges_in_core << std::endl;
} else{
  // hash vert in core
  pbbslib::sparse_additive_map<gbbs::uintE, gbbs::uintE> freq_cores = pbbslib::sparse_additive_map<gbbs::uintE, gbbs::uintE>(graph_.Graph()->n, std::make_tuple<gbbs::uintE, gbbs::uintE>(UINT_E_MAX, UINT_E_MAX));
  pbbs::parallel_for(0, graph_.Graph()->n, [&](std::size_t i){
    freq_cores.insert(std::make_tuple(cores[i], 1));
  });
  auto freq_cores_table = freq_cores.entries();
  freq_cores_table = pbbslib::sample_sort(freq_cores_table, [&](std::tuple<gbbs::uintE, gbbs::uintE> a, std::tuple<gbbs::uintE, gbbs::uintE> b){
    return std::get<0>(a) > std::get<0>(b);
  }, true);
  /*gbbs::uintE sum = 0;
  for (std::size_t i = 1; i < freq_cores_table.size(); i ++) {
    if(std::get<0>(freq_cores_table[i-1]) <= std::get<0>(freq_cores_table[i])) std::cout << "ERR" << std::endl;
    sum += std::get<1>(freq_cores_table[i]);
  }
  sum += std::get<1>(freq_cores_table[0]);
  std::cout << " Sum: " << sum << std::endl;
  assert(sum == graph_.Graph()->n);*/
  //std::cout << "First: " << std::get<0>(freq_cores_table[freq_cores_table.size() - 1]) << std::endl;
  //std::cout << "Last: " << std::get<0>(freq_cores_table[0]) << std::endl;
  //assert(std::get<0>(freq_cores_table[freq_cores_table.size() - 1]) <= std::get<0>(freq_cores_table[0]));
  // now do a prefix sum, then a percentage
  /*double cut = config.kcore_config().percent_cutoff() * graph_.Graph()->n;
  double sofar = 0;
  for (std::size_t i = 0; i < freq_cores_table.size(); i ++) {
    sofar += std::get<1>(freq_cores_table[i]);
    if (sofar >= cut) {
      cutoff = std::get<0>(freq_cores_table[i]);
      break;
    }
  }
  std::cout << "Num vert; " << gbbs::uintE{config.kcore_config().percent_cutoff() * graph_.Graph()->n} << ", " << graph_.Graph()->n << std::endl;
  std::cout << "Sofar: " << sofar << std::endl;*/
  auto add_mon = [](std::tuple<gbbs::uintE, gbbs::uintE> a, std::tuple<gbbs::uintE, gbbs::uintE> b){
    return std::make_tuple(std::get<0>(b), std::get<1>(a) + std::get<1>(b));
  };
  auto mon = pbbslib::make_monoid(add_mon, std::make_tuple(gbbs::uintE{0}, gbbs::uintE{0}));
  auto total = pbbs::scan_inplace(freq_cores_table.slice(), mon);
  // find the right percent cutoff in the prefix sum
  auto found_idx = pbbslib::binary_search(freq_cores_table, 
    std::make_tuple(gbbs::uintE{0}, gbbs::uintE{config.kcore_config().percent_cutoff() * graph_.Graph()->n}),
    [](std::tuple<gbbs::uintE, gbbs::uintE> a, std::tuple<gbbs::uintE, gbbs::uintE> b){
      return std::get<1>(a) < std::get<1>(b);
    });
  cutoff = found_idx != freq_cores_table.size() ? std::get<0>(freq_cores_table[found_idx]) : 
    std::get<0>(freq_cores_table[freq_cores_table.size() - 1]);
  
  /*std::cout << "num: " << std::get<1>(freq_cores_table[found_idx]) << std::endl;
  std::cout << "num after: " << std::get<1>(freq_cores_table[found_idx + 1]) << std::endl;
  std::cout << "core after: " << std::get<0>(freq_cores_table[found_idx + 1]) << std::endl;
  std::cout << "num after: " << std::get<1>(freq_cores_table[found_idx - 1]) << std::endl;
  std::cout << "core after: " << std::get<0>(freq_cores_table[found_idx - 1]) << std::endl;*/
  std::cout << "Cutoff core: " << cutoff << std::endl;
}
      // Prune graph to be only vert in higher cores -- > 7 maybe
      auto pack_predicate = [&](const gbbs::uintE& u, const gbbs::uintE& v, const float& wgh) {
        return (cores[u] >= cutoff && cores[v] >= cutoff);
      };
      auto G_core = filterGraph(*(graph_.Graph()), pack_predicate);
      // Run RefineClusters on the sparsified graph
      // **** TODO: here we need to fix it so that the config is the modularity we want
      // std::tuple<std::vector<double>, double, std::size_t> ComputeModularityConfig(
  //const gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* graph, double resolution)
      auto new_config_params = ComputeModularityConfig(&G_core, original_resolution);
      ClustererConfig core_config;
      core_config.CopyFrom(clusterer_config);
      core_config.mutable_correlation_clusterer_config()->set_resolution(std::get<1>(new_config_params));
      core_config.mutable_correlation_clusterer_config()->set_edge_weight_offset(0);
      core_config.mutable_correlation_clusterer_config()->mutable_louvain_config()->set_num_inner_iterations(5);
      core_config.mutable_correlation_clusterer_config()->mutable_louvain_config()->set_num_iterations(2);
      RETURN_IF_ERROR(RefineClusters_subroutine(core_config, initial_clustering, std::get<0>(new_config_params), &G_core));
//}
      //std::cout << "START COMPRESS" << std::endl;
      // Now, redo clustering on original graph
      // This way does not force original clusters to be maintained
      if (config.kcore_config().fix_preclusters()) {
      std::vector<gbbs::uintE> cluster_ids(graph_.Graph()->n);
      std::vector<gbbs::uintE> local_cluster_ids(graph_.Graph()->n);
      pbbs::parallel_for(0, initial_clustering->size(), [&](std::size_t i) {
        for (std::size_t j = 0; j < (*initial_clustering)[i].size(); j++) {
          cluster_ids[(*initial_clustering)[i][j]] = i;
        }
      });
      auto helper = absl::make_unique<ClusteringHelper>(graph_.Graph()->n,
        clusterer_config, node_weights, *initial_clustering);
      GraphWithWeights new_compressed_graph;
      ASSIGN_OR_RETURN(
          new_compressed_graph,
          CompressGraph(*(graph_.Graph()), cluster_ids, helper.get()));
      InMemoryClusterer::Clustering local_clustering(new_compressed_graph.graph->n);
      // Create all-singletons initial clustering
      pbbs::parallel_for(0, new_compressed_graph.graph->n, [&](std::size_t i) {
        local_clustering[i] = {static_cast<int32_t>(i)};
      });
      //std::cout << "START NEXT REFINE" << std::endl;
      RETURN_IF_ERROR(RefineClusters_subroutine(clusterer_config, &local_clustering,
        new_compressed_graph.node_weights, new_compressed_graph.graph.get()));
      //std::cout << "END NEXT REFINE" << std::endl;
      pbbs::parallel_for(0, local_clustering.size(), [&](std::size_t i) {
        for (std::size_t j = 0; j < local_clustering[i].size(); j++) {
          local_cluster_ids[local_clustering[i][j]] = i;
        }
      });
      cluster_ids = FlattenClustering(cluster_ids, local_cluster_ids);
      auto get_clusters = [&](NodeId i) -> NodeId { return i; };
      *initial_clustering = parallel::OutputIndicesById<ClusterId, NodeId>(
        cluster_ids, get_clusters, cluster_ids.size());
      } else {
        RETURN_IF_ERROR(RefineClusters_subroutine(clusterer_config, initial_clustering, node_weights, graph));
      }
   } else {
      RETURN_IF_ERROR(RefineClusters_subroutine(clusterer_config, initial_clustering, node_weights, graph));
  }
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
