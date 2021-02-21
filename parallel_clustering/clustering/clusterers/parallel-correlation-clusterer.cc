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

#include "external/gbbs/benchmarks/KCore/JulienneDBS17/KCore.h"
#include "external/gbbs/gbbs/pbbslib/sparse_additive_map.h"
#include "external/gbbs/pbbslib/random_shuffle.h"
//#include "external/gbbs/pbbslib/union_find.h"

namespace research_graph {
namespace in_memory {

absl::Status ParallelCorrelationClusterer::RefineClusters(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering) const {
  std::vector<double> empty;
  return RefineClusters(clusterer_config, initial_clustering, empty);
}

absl::Status ParallelCorrelationClusterer::RefineClusters(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    gbbs::symmetric_graph<gbbs::csv_bytepd_amortized, pbbslib::empty>* graph) const {
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
    std::vector<double> node_weights) const {
    return RefineClusters_subroutine(clusterer_config, initial_clustering, node_weights, graph_.Graph());
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
    gbbs::symmetric_graph<gbbs::csv_bytepd_amortized, pbbslib::empty>* graph, double original_resolution) const {
  const auto& config = clusterer_config.correlation_clusterer_config();

      RETURN_IF_ERROR(RefineClusters_subroutine(clusterer_config, initial_clustering, node_weights, graph));
  return absl::OkStatus();
}

absl::StatusOr<InMemoryClusterer::Clustering>
ParallelCorrelationClusterer::Cluster(
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
