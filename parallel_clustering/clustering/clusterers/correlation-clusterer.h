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

#ifndef PARALLEL_CLUSTERING_CLUSTERERS_CORRELATION_CLUSTERER_H_
#define PARALLEL_CLUSTERING_CLUSTERERS_CORRELATION_CLUSTERER_H_

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>
#include <iomanip>
#include <iostream>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "clustering/clusterers/correlation-clusterer-internal.h"
#include "clustering/config.pb.h"
#include "clustering/gbbs-graph.h"
#include "clustering/in-memory-clusterer.h"
#include "parallel/parallel-graph-utils.h"
#include "clustering/status_macros.h"


#include "clustering/clusterers/parallel-correlation-clusterer-internal.h"


#include "external/gbbs/gbbs/pbbslib/sparse_additive_map.h"


namespace research_graph {
namespace in_memory {

// A local-search based clusterer optimizing the correlation clustering
// objective. See comment above CorrelationClustererConfig in
// ../config.proto for more. This uses the CorrelationClustererConfig proto.
// Also, note that the input graph is required to be undirected.
class CorrelationClusterer : public InMemoryClusterer {
 public:
  using ClusterId = gbbs::uintE;

  Graph* MutableGraph() override { return &graph_; }

  absl::StatusOr<Clustering> Cluster(
      const ClustererConfig& config) const override;

  // initial_clustering must include every node in the range
  // [0, MutableGraph().NumNodes()) exactly once.
  absl::Status RefineClusters(const ClustererConfig& clusterer_config,
                              Clustering* initial_clustering) const override;
  
  absl::Status RefineClusters(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    gbbs::symmetric_graph<gbbs::csv_bytepd_amortized, pbbslib::empty>* graph) const;

  absl::Status RefineClusters(const ClustererConfig& clusterer_config,
                              Clustering* initial_clustering,
                              std::vector<double> node_weights, double resolution = 0) const;

  absl::Status RefineClusters(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    std::vector<double> node_weights,
    gbbs::symmetric_graph<gbbs::csv_bytepd_amortized, pbbslib::empty>* graph, double original_resolution = 0) const;

  absl::Status RefineClusters_subroutine(const ClustererConfig& clusterer_config,
                              Clustering* initial_clustering) const;
  
  template<class G>
  absl::Status RefineClusters_subroutine(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    G* graph) const;

  absl::Status RefineClusters_subroutine(const ClustererConfig& clusterer_config,
                              Clustering* initial_clustering,
                              std::vector<double> node_weights) const;
  
  template<class G>
  absl::Status RefineClusters_subroutine(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    std::vector<double> node_weights,
    G* graph) const;

 protected:
  GbbsGraph graph_;
};


namespace {

std::vector<std::vector<gbbs::uintE>> SeqOutputIndicesById(
    const std::vector<gbbs::uintE>& index_ids,
    const std::function<gbbs::uintE(gbbs::uintE)>& get_indices_func, gbbs::uintE num_indices) {
      using A = gbbs::uintE;
      using B = gbbs::uintE;
  if (num_indices == 0) {
    std::vector<std::vector<B>> finished_indices;
    return finished_indices;
  }
  // Sort all vertices by cluster id
  std::vector<B> indices_sort(num_indices);
  for (std::size_t i = 0; i < num_indices; i++) {
    indices_sort[i] = get_indices_func(i);
  }
  std::sort(indices_sort.begin(), indices_sort.end(), [&](B a, B b) -> bool{index_ids[a] < index_ids[b];});

  // Boundary indices indicate sections corresponding to clusters
  std::vector<std::vector<B>> finished_indices(1);
  finished_indices[0].push_back(indices_sort[0]);
  auto prev = index_ids[indices_sort[0]];
  for (std::size_t i = 1; i < num_indices; i++) {
    if (index_ids[indices_sort[i]] != prev) {
      prev = index_ids[indices_sort[i]];
      finished_indices.push_back(std::vector<B>());
    }
    finished_indices[finished_indices.size()-1].push_back(indices_sort[i]);
  }

  return finished_indices;
}

std::vector<gbbs::uintE> SeqFlattenClustering(
    const std::vector<gbbs::uintE>& cluster_ids,
    const std::vector<gbbs::uintE>& compressed_cluster_ids) {
  std::vector<gbbs::uintE> new_cluster_ids(cluster_ids.size());
  for (std::size_t i = 0; i < cluster_ids.size(); i++) {
    new_cluster_ids[i] = (cluster_ids[i] == UINT_E_MAX)
                             ? UINT_E_MAX
                             : compressed_cluster_ids[cluster_ids[i]];
  }
  return new_cluster_ids;
}

struct CorrelationClustererRefine2 {
  using H = std::unique_ptr<SeqClusteringHelper>;
  using GX = std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>;
  gbbs::sequence<H> recurse_helpers = gbbs::sequence<H>(0, [](std::size_t i){return H(nullptr);});
  gbbs::sequence<G> recurse_graphs = gbbs::sequence<GX>(0, [](std::size_t i){return G(nullptr);});
  bool use_refine = false;
};

// Given a vertex subset moved_subset, computes best moves for all vertices
// and performs the moves. Returns a vertex subset consisting of all vertices
// adjacent to modified clusters.
template<class G>
std::vector<gbbs::uintE>
BestMovesForVertexSubset(
    G* current_graph,
    std::size_t num_nodes, std::vector<gbbs::uintE>& moved_subset,
    SeqClusteringHelper* helper, const ClustererConfig& clusterer_config) {
  // Find best moves per vertex in moved_subset
  std::vector<bool> moved_vertex = std::vector<bool>(current_graph->n, false);
  std::vector<bool> moved_clusters = std::vector<bool>(current_graph->n, false);

  for (std::size_t i = 0; i < moved_subset.size(); i++) {
    moved_vertex[moved_subset[i]] = helper->AsyncMove(*current_graph, moved_subset[i]);
    if (moved_vertex[moved_subset[i]]) {
      moved_clusters[helper->ClusterIds()[moved_subset[i]]] = true;
    }
  }

  if (clusterer_config.correlation_clusterer_config().move_method() == CorrelationClustererConfig::ALL_MOVE) {
    auto new_moved_subset = std::vector<gbbs::uintE>(num_nodes);
    for (std::size_t i = 0; i < num_nodes; i++) {
      new_moved_subset[i] = i;
    }
    return new_moved_subset;
  }

  auto new_moved_subset = std::vector<gbbs::uintE>();
  new_moved_subset.reserve(num_nodes);
  auto affected_vertices = std::vector<gbbs::uintE>();
  affected_vertices.reserve(num_nodes);

  bool nbhr_cluster_move = clusterer_config.correlation_clusterer_config().move_method() == CorrelationClustererConfig::NBHR_CLUSTER_MOVE;
  if (nbhr_cluster_move) {
    for (std::size_t i = 0; i < num_nodes; i++) {
      if (moved_clusters[helper->ClusterIds()[i]]) affected_vertices.push_back(i);
    }
  } else {
    for (std::size_t i = 0; i < moved_subset.size(); i++) {
      if (moved_vertex[moved_subset[i]]) affected_vertices.push_back(moved_subset[i]);
    }
  }

  std::vector<bool> affected_neighbors = std::vector<bool>(current_graph->n, false);
    // Mark in moved vertex adjacent vertices
    for (std::size_t i = 0; i < affected_vertices.size(); i++) {
      auto vtx = current_graph->get_vertex(affected_vertices[i]);
      for (std::size_t j = 0; j < vtx.getOutDegree(); j++) {
        auto nbhr = vtx.getOutNeighbor(j);
        affected_neighbors[nbhr] = true;
      }
    }
    // translate affected neighbors to new_moved_subset
    for (std::size_t i = 0; i < affected_neighbors.size(); i++) {
      if (affected_neighbors[i]) new_moved_subset.push_back(i);
    }
    return new_moved_subset;
}

template<class G>
bool IterateBestMoves(int num_inner_iterations, const ClustererConfig& clusterer_config,
  G* current_graph,
  SeqClusteringHelper* helper) {
  const auto num_nodes = current_graph->n;
  bool moved = false;
  bool local_moved = true;
  auto moved_subset = std::vector<gbbs::uintE>(num_nodes);
  for (std::size_t i = 0; i < num_nodes; i++) {
    moved_subset[i] = i;
  }

  // Iterate over best moves
  int local_iter = 0;
  for (local_iter = 0; local_iter < num_inner_iterations && local_moved; ++local_iter) {
    auto new_moved_subset =
      BestMovesForVertexSubset(current_graph, num_nodes, moved_subset,
                              helper, clusterer_config);
    moved_subset.swap(new_moved_subset);
    local_moved = !moved_subset.empty();
    moved |= local_moved;
  }
  std::cout << "Num inner: " << local_iter << std::endl;
  return moved;
}

}  // namespace

template<class G>
absl::Status CorrelationClusterer::RefineClusters_subroutine(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
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
  CorrelationClustererRefine2 refine{};
  if (config.refine()) {
    using H = std::unique_ptr<SeqClusteringHelper>;
    using GX = std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>;
    refine.recurse_helpers = gbbs::sequence<H>(num_iterations);
    refine.recurse_graphs = gbbs::sequence<GX>(num_iterations);
    for (std::size_t i = 0; i < num_iterations; i++) {
      refine.recurse_helpers[i] = H(nullptr);
      refine.recurse_graphs[i] = G(nullptr);
    }
  }

  std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>
      compressed_graph;

  // Initialize clustering helper
  auto helper = node_weights.empty() ? absl::make_unique<SeqClusteringHelper>(
      graph->n, clusterer_config, *initial_clustering) :
      absl::make_unique<SeqClusteringHelper>(
      graph->n, clusterer_config, node_weights, *initial_clustering);
  // The max objective is the maximum objective given by the inner iterations
  // of best moves rounds

  std::vector<gbbs::uintE> cluster_ids(graph->n);
  std::vector<gbbs::uintE> local_cluster_ids(graph->n);
  for (std::size_t i = 0; i < graph->n; i++) {
    cluster_ids[i] = i;
  }

  int iter = 0;
  for (iter = 0; iter < num_iterations; ++iter) {
    //gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph =
    //    (iter == 0) ? graph : compressed_graph.get();

    // Initialize subclustering data structure
    bool moved = false;
    if (iter==0) moved = IterateBestMoves(num_inner_iterations, clusterer_config, graph,
      helper.get());
      else moved = IterateBestMoves(num_inner_iterations, clusterer_config, compressed_graph.get(),
      helper.get());

    // If no moves can be made at all, exit
    if (!moved) {
      iter--;
      break;
    }

    // Compress cluster ids in initial_helper based on helper
    if (!config.refine()) cluster_ids = SeqFlattenClustering(cluster_ids, helper->ClusterIds());
    else if (config.refine() && iter == num_iterations - 1) {
      refine.recurse_helpers[iter] = std::move(helper);
      refine.recurse_graphs[iter] = (iter == 0) ? nullptr : std::move(compressed_graph);
    } 

    if (iter == num_iterations - 1) break;

    // Compress graph
    auto n = (iter == 0) ? graph->n : compressed_graph->n;
    GraphWithWeights new_compressed_graph;
    InMemoryClusterer::Clustering new_clustering{};
    for (std::size_t i = 0; i < n; i++) {
        local_cluster_ids[i] = helper->ClusterIds()[i];
    }

    if (iter == 0) {
      ASSIGN_OR_RETURN(
          new_compressed_graph,
          SeqCompressGraph(*graph, local_cluster_ids, helper.get()));
    } else {
ASSIGN_OR_RETURN(
          new_compressed_graph,
          SeqCompressGraph(*(compressed_graph.get()), local_cluster_ids, helper.get()));
    }
      // Create new local clusters
    for (std::size_t i = 0; i < new_compressed_graph.graph->n; i++) {
      local_cluster_ids[i] = i;
    }

    compressed_graph.swap(new_compressed_graph.graph);
    if (config.refine()) {
      refine.recurse_helpers[iter] = std::move(helper);
      refine.recurse_graphs[iter] = std::move(new_compressed_graph.graph);
    } else if (new_compressed_graph.graph) new_compressed_graph.graph->del();

    helper = absl::make_unique<SeqClusteringHelper>(
        compressed_graph->n, clusterer_config,
        new_compressed_graph.node_weights, new_clustering);
  }

  // Refine clusters up the stack
  if (config.refine() && iter > 0) {
    auto get_clusters = [&](NodeId i) -> NodeId { return i; };
    for (int i = iter - 1; i >= 0; i--) {
      //gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph =
      //  (i == 0) ? graph : refine.recurse_graphs[i].get();

      auto flatten_cluster_ids = SeqFlattenClustering(refine.recurse_helpers[i]->ClusterIds(),
          refine.recurse_helpers[i+1]->ClusterIds());
      auto flatten_clustering = SeqOutputIndicesById(
          flatten_cluster_ids,
          get_clusters, 
          flatten_cluster_ids.size());
      refine.recurse_helpers[i]->ResetClustering(flatten_clustering);
      //std::cout << "REFINE2: " << i << std::endl;
      // Take iter + 1 helper; use it to compress into iter helper; then
      // do best moves again, using the graph and node weights from iter

      // TODO: get rid of subclustering here
      if (i == 0) {
      IterateBestMoves(num_inner_iterations, clusterer_config, graph,
        refine.recurse_helpers[i].get());
      }
      else {
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

  *initial_clustering = SeqOutputIndicesById(
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



  return absl::OkStatus();
}

template<class G>
absl::Status CorrelationClusterer::RefineClusters_subroutine(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    G* graph) const {
  std::vector<double> empty;
  return RefineClusters_subroutine(clusterer_config, initial_clustering, empty, graph);
}


}  // namespace in_memory
}  // namespace research_graph

#endif  // PARALLEL_CLUSTERING_CLUSTERERS_CORRELATION_CLUSTERER_H_
