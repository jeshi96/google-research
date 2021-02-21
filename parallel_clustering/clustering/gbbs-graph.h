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

#ifndef RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_GBBS_GRAPH_H_
#define RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_GBBS_GRAPH_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "clustering/in-memory-clusterer.h"
#include "external/gbbs/gbbs/gbbs.h"
#include "external/gbbs/gbbs/graph.h"
#include "external/gbbs/gbbs/macros.h"

namespace research_graph {
namespace in_memory {

// Represents a weighted undirected graph in GBBS format.
// Multiple edges and self-loops are allowed.
// Note that GBBS doesn't support node weights.
// Also, Import does not automatically symmetrize the graph. If a vertex u is in
// the adjacency list of a vertex v, then it is not guaranteed that vertex v
// will appear in the adjacency list of vertex u unless explicitly
// specified in vertex u's adjacency list.
template<class ClusterGraph>
class GbbsGraph : public InMemoryClusterer<ClusterGraph>::Graph {
 public:
  ClusterGraph* Graph() const{
  return graph_.get();
}

  //std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>
  //    graph_;

 private:
  absl::Mutex mutex_;
};

}  // namespace in_memory
}  // namespace research_graph

#endif  // RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_GBBS_GRAPH_H_
