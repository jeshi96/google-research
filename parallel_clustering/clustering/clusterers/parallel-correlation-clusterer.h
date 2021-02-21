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

#ifndef PARALLEL_CLUSTERING_CLUSTERERS_PARALLEL_CORRELATION_CLUSTERER_H_
#define PARALLEL_CLUSTERING_CLUSTERERS_PARALLEL_CORRELATION_CLUSTERER_H_

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>
#include <iomanip>
#include <iostream>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "clustering/clusterers/parallel-correlation-clusterer-internal.h"
#include "clustering/config.pb.h"
#include "clustering/gbbs-graph.h"
#include "clustering/in-memory-clusterer.h"
#include "parallel/parallel-graph-utils.h"
#include "clustering/status_macros.h"

namespace research_graph {
namespace in_memory {

  using NodeId  = gbbs::uintE;

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
template <class Graph>
std::unique_ptr<gbbs::vertexSubset, void (*)(gbbs::vertexSubset*)>
BestMovesForVertexSubset(
    Graph* current_graph,
    std::size_t num_nodes, gbbs::vertexSubset* moved_subset,
    ClusteringHelper* helper, const ClustererConfig& clusterer_config) {
  bool async = clusterer_config.correlation_clusterer_config().async();
  std::vector<absl::optional<ClusteringHelper::ClusterId>> moves(num_nodes,
                                                                 absl::nullopt);
  std::vector<char> moved_vertex(num_nodes, 0);

  // Find best moves per vertex in moved_subset
  gbbs::sequence<bool> async_mark = gbbs::sequence<bool>(current_graph->n, false);
  auto moved_clusters = absl::make_unique<bool[]>(current_graph->n);
  pbbs::parallel_for(0, current_graph->n,
                     [&](std::size_t i) { moved_clusters[i] = false; });
  gbbs::vertexMap(*moved_subset, [&](std::size_t i) {
  //for (std::size_t i = 0; i < current_graph->n; i++) {
    if (async) {
      auto move = helper->AsyncMove(*current_graph, i);
      if (move) {
        pbbslib::CAS<bool>(&moved_clusters[helper->ClusterIds()[i]], false, true);
        moved_vertex[i] = 1;
      }
    }
  });

  if (clusterer_config.correlation_clusterer_config().move_method() == CorrelationClustererConfig::ALL_MOVE) {
    return std::unique_ptr<gbbs::vertexSubset, void (*)(gbbs::vertexSubset*)>(
    new gbbs::vertexSubset(num_nodes, num_nodes,
    gbbs::sequence<bool>(num_nodes, true).to_array()),
    [](gbbs::vertexSubset* subset) {
      subset->del();
      delete subset;
    });
  }

  bool default_move = clusterer_config.correlation_clusterer_config().move_method() == CorrelationClustererConfig::NBHR_CLUSTER_MOVE;
  // Mark vertices adjacent to clusters that have moved; these are
  // the vertices whose best moves must be recomputed.
  auto local_moved_subset =
      std::unique_ptr<gbbs::vertexSubset, void (*)(gbbs::vertexSubset*)>(
          new gbbs::vertexSubset(
              num_nodes, num_nodes,
              gbbs::sequence<bool>(
                  num_nodes,
                  [&](std::size_t i) {
                    if (default_move)
                      return moved_clusters[helper->ClusterIds()[i]];
                    else
                      return (bool) moved_vertex[i];
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

template <class Graph>
bool IterateBestMoves(int num_inner_iterations, const ClustererConfig& clusterer_config,
  Graph* current_graph,
  ClusteringHelper* helper) {
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
                              helper, clusterer_config);
    moved_subset.swap(new_moved_subset);
    local_moved = !moved_subset->isEmpty();
    moved |= local_moved;
  }
  std::cout << "Num inner: " << local_iter << std::endl;
  return moved;
}
}  // namespace

using Clustering = std::vector<std::vector<gbbs::uintE>>;

// A local-search based clusterer optimizing the correlation clustering
// objective. See comment above CorrelationClustererConfig in
// ../config.proto for more. This uses the CorrelationClustererConfig proto.
// Also, note that the input graph is required to be undirected.
template<class ClusterGraph>
class ParallelCorrelationClusterer : public InMemoryClusterer<ClusterGraph>{
  public:
  GbbsGraph<ClusterGraph> graph_;

  GbbsGraph<ClusterGraph>* MutableGraph() override { return &graph_; }
 
  using ClusterId = gbbs::uintE;



  // initial_clustering must include every node in the range
  // [0, MutableGraph().NumNodes()) exactly once.

  template<class G>
  absl::Status RefineClusters_subroutine(
    const ClustererConfig& clusterer_config,
    Clustering* initial_clustering,
    std::vector<double> node_weights,
    G* graph) const {
pbbs::timer t; t.start();
      //std::cout << "REFINE" << std::endl;
      //fflush(stdout);
    const auto& config = clusterer_config.correlation_clusterer_config();
  // Set number of iterations based on clustering method
  int num_iterations = 0;
  int num_inner_iterations = 0;
  switch (config.clustering_moves_method()) {
    case CorrelationClustererConfig::DEFAULT_CLUSTER_MOVES:
      num_iterations = 1;
      num_inner_iterations =
          config.num_iterations() > 0 ? config.num_iterations() : 20;
      break;
    case CorrelationClustererConfig::LOUVAIN:
      num_iterations = config.louvain_config().num_iterations() > 0
                           ? config.louvain_config().num_iterations()
                           : 10;
      num_inner_iterations =
          config.louvain_config().num_inner_iterations() > 0
              ? config.louvain_config().num_inner_iterations()
              : 10;
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
    using GX = std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>;
    refine.recurse_helpers = gbbs::sequence<H>(num_iterations, [](std::size_t i){return H(nullptr);});
    refine.recurse_graphs = gbbs::sequence<GX>(num_iterations, [](std::size_t i){return G(nullptr);});
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
    auto n = (iter == 0) ? graph->n : compressed_graph->n;

    // Initialize subclustering data structure
    bool moved = false;
    if (iter == 0) moved = IterateBestMoves(num_inner_iterations, clusterer_config, graph, helper.get());
    else moved = IterateBestMoves(num_inner_iterations, clusterer_config, compressed_graph.get(), helper.get());

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
    Clustering new_clustering{};
    pbbs::parallel_for(0, n, [&](std::size_t i) {
        local_cluster_ids[i] = helper->ClusterIds()[i];
    });
    if (iter == 0) {
      ASSIGN_OR_RETURN(
          new_compressed_graph,
          CompressGraph(*graph, local_cluster_ids, helper.get()));
    } else {
      ASSIGN_OR_RETURN(
          new_compressed_graph,
          CompressGraph(*(compressed_graph.get()), local_cluster_ids, helper.get()));
    }
      // Create new local clusters
      pbbs::parallel_for(0, new_compressed_graph.graph->n,
                         [&](std::size_t i) { local_cluster_ids[i] = i; });

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
  if (config.refine() && iter > 0) {
    auto get_clusters = [&](NodeId i) -> NodeId { return i; };
    for (int i = iter - 1; i >= 0; i--) {

      auto flatten_cluster_ids = FlattenClustering(refine.recurse_helpers[i]->ClusterIds(),
          refine.recurse_helpers[i+1]->ClusterIds());
      auto flatten_clustering = parallel::OutputIndicesById<ClusterId, NodeId>(
          flatten_cluster_ids,
          get_clusters, 
          flatten_cluster_ids.size());
      refine.recurse_helpers[i]->ResetClustering(flatten_clustering);


      // TODO: get rid of subclustering here
      if (i == 0) {
      IterateBestMoves(num_inner_iterations, clusterer_config, graph,
        refine.recurse_helpers[i].get());
      } else {
        IterateBestMoves(num_inner_iterations, clusterer_config, refine.recurse_graphs[i].get(),
        refine.recurse_helpers[i].get());
      }
    }
    cluster_ids = refine.recurse_helpers[0]->ClusterIds();
  }
  

t.stop(); t.reportTotal("Actual Cluster Time: ");
  std::cout << "Num outer: " << iter << std::endl;

  if (compressed_graph) compressed_graph->del();

  auto get_clusters = [&](NodeId i) -> NodeId { return i; };

  *initial_clustering = parallel::OutputIndicesById<ClusterId, NodeId>(
      cluster_ids, get_clusters, cluster_ids.size());

  // Hack to compute objective
  auto helper2 = node_weights.empty() ? absl::make_unique<ClusteringHelper>(
      graph->n, clusterer_config, *initial_clustering) :
      absl::make_unique<ClusteringHelper>(
      graph->n, clusterer_config, node_weights, *initial_clustering);
  // The max objective is the maximum objective given by the inner iterations
  // of best moves rounds
  double max_objective = helper2->ComputeObjective(*graph);
  std::cout << "Objective: " << std::setprecision(17) << max_objective << std::endl;

  // Now, we must compute the disagreement objective
  double max_disagreement_objective = helper2->ComputeDisagreementObjective(*graph);
  std::cout << "Disagreement Objective: " << std::setprecision(17) << max_disagreement_objective << std::endl;

  std::cout << "Number of Clusters: " << initial_clustering->size() << std::endl;

  if (config.connect_stats()) {
    auto num_disconnect = NumDisconnected(*initial_clustering, graph);
    std::cout << "Number of Disconnected Clusters: " << num_disconnect << std::endl;
  }

  return absl::OkStatus();
}

  template<class Graph>
  absl::Status RefineClusters_subroutine(
    const ClustererConfig& clusterer_config,
    Clustering* initial_clustering,
    Graph* graph) {
  std::vector<double> empty;
  return RefineClusters_subroutine(clusterer_config, initial_clustering, empty, graph);
}


  absl::Status RefineClusters_subroutine(const ClustererConfig& clusterer_config,
                              Clustering* initial_clustering,
                              std::vector<double> node_weights) const {
    return RefineClusters_subroutine(clusterer_config, initial_clustering, node_weights, graph_.Graph());
}


  absl::Status RefineClusters_subroutine(const ClustererConfig& clusterer_config,
                              Clustering* initial_clustering) const {
  std::vector<double> empty;
  return RefineClusters_subroutine(clusterer_config, initial_clustering, empty);
}


template<class G>
  absl::Status RefineClusters(
    const ClustererConfig& clusterer_config,
    Clustering* initial_clustering,
    std::vector<double> node_weights,
    G* graph, double original_resolution = 0) const {
  const auto& config = clusterer_config.correlation_clusterer_config();
      RETURN_IF_ERROR(RefineClusters_subroutine(clusterer_config, initial_clustering, node_weights, graph));
  return absl::OkStatus();
}

  absl::Status RefineClusters(const ClustererConfig& clusterer_config,
                              Clustering* initial_clustering,
                              std::vector<double> node_weights, double resolution = 0) const {
    return RefineClusters(clusterer_config, initial_clustering, node_weights, graph_.Graph(), resolution);
}

  template<class G>
    absl::Status RefineClusters(
    const ClustererConfig& clusterer_config,
    Clustering* initial_clustering,
    G* graph) const{
  std::vector<double> empty;
  return RefineClusters(clusterer_config, initial_clustering, empty, graph);
}

  absl::Status RefineClusters(const ClustererConfig& clusterer_config,
                              Clustering* initial_clustering) const override {
  std::vector<double> empty;
  return RefineClusters(clusterer_config, initial_clustering, empty);
  }

  absl::StatusOr<Clustering> Cluster(
      const ClustererConfig& clusterer_config) const override{
  Clustering clustering(graph_.Graph()->n);
  // Create all-singletons initial clustering
  pbbs::parallel_for(0, graph_.Graph()->n, [&](std::size_t i) {
    clustering[i] = {static_cast<gbbs::uintE>(i)};
  });

  RETURN_IF_ERROR(RefineClusters(clusterer_config, &clustering));

  return clustering;
}


};

}  // namespace in_memory
}  // namespace research_graph

#endif  // PARALLEL_CLUSTERING_CLUSTERERS_PARALLEL_CORRELATION_CLUSTERER_H_
