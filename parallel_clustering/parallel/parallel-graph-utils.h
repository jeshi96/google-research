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

#ifndef RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PARALLEL_GRAPH_UTILS_H_
#define RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PARALLEL_GRAPH_UTILS_H_

#include <cstdio>

#include "external/gbbs/gbbs/gbbs.h"
#include "external/gbbs/gbbs/graph_io.h"
#include "external/gbbs/gbbs/macros.h"
#include "external/gbbs/pbbslib/seq.h"
#include "external/gbbs/pbbslib/sequence_ops.h"
#include "external/gbbs/pbbslib/utilities.h"
#include "parallel/parallel-sequence-ops.h"

namespace research_graph {

namespace {

float FloatFromWeightPGU(float weight) { return weight; }
float FloatFromWeightPGU(pbbslib::empty weight) { return 1; }

}

// Retrieves a list of inter-cluster edges, given a set of cluster_ids
// that form the vertices of a new graph. Maps all edges in original_graph
// to cluster ids. Depending on is_valid_func, combines edges in the same
// cluster to be a self-loop.
template <class Graph>
std::vector<std::tuple<gbbs::uintE, gbbs::uintE, float>>
RetrieveInterClusterEdges(
    Graph& original_graph,
    const std::vector<gbbs::uintE>& cluster_ids,
    const std::function<bool(gbbs::uintE, gbbs::uintE)>& is_valid_func) {
  using W = typename Graph::weight_type;
  // First, compute offsets on the original graph
  std::vector<gbbs::uintE> all_offsets(original_graph.n + 1, gbbs::uintE{0});
  pbbs::parallel_for(0, original_graph.n, [&](std::size_t i) {
    all_offsets[i] = original_graph.get_vertex(i).getOutDegree();
  });
  std::pair<pbbs::sequence<gbbs::uintE>, gbbs::uintE> all_offsets_scan =
      research_graph::parallel::ScanAdd(absl::Span<const gbbs::uintE>(
          all_offsets.data(), all_offsets.size()));

  // Retrieve all edges in the graph, mapped to cluster_ids
  std::vector<std::tuple<gbbs::uintE, gbbs::uintE, float>> all_edges(
      all_offsets_scan.second, std::make_tuple(UINT_E_MAX, UINT_E_MAX, float{0}));

  pbbs::parallel_for(0, original_graph.n, [&](std::size_t j) {
    auto vtx = original_graph.get_vertex(j);
    gbbs::uintE i = 0;
    if (cluster_ids[j] != UINT_E_MAX) {
      auto map_f = [&](gbbs::uintE u, gbbs::uintE v, W w) {
        float weight = FloatFromWeightPGU(w);
        if (is_valid_func(cluster_ids[v], cluster_ids[u]) &&
            cluster_ids[v] != UINT_E_MAX 
            // TODO: they do this line, which doesn't double count self loops
            // but somehow if we do double count self loops, it makes us match
            // their modularity better?
            // ah our volume of a supernode is off depending on this --
            // volume is taken to be the sum, which does double count self loops
            // so if we want to match PLM, we have to subtract weight of
            // self loop from weight of supernode, and put this line back in.
            && (v <= u || cluster_ids[v] != cluster_ids[u])
            )
          all_edges[all_offsets_scan.first[j] + i] =
              std::make_tuple(cluster_ids[u], cluster_ids[v], weight);
        i++;
      };
      vtx.mapOutNgh(j, map_f, false);
    }
  });

  // Filter for valid edges
  std::vector<std::tuple<gbbs::uintE, gbbs::uintE, float>> filtered_edges =
      research_graph::parallel::FilterOut<
          std::tuple<gbbs::uintE, gbbs::uintE, float>>(
          absl::Span<const std::tuple<gbbs::uintE, gbbs::uintE, float>>(
              all_edges.data(), all_offsets_scan.second),
          [](std::tuple<gbbs::uintE, gbbs::uintE, float> x) {
            return std::get<0>(x) != UINT_E_MAX && std::get<1>(x) != UINT_E_MAX;
          });
  return filtered_edges;
}

struct OffsetsEdges {
  std::vector<gbbs::uintE> offsets;
  std::unique_ptr<std::tuple<gbbs::uintE, float>[]> edges;
  std::size_t num_edges;
};

// Given get_key, which is nondecreasing, defined for 0, ..., num_keys-1, and
// returns an unsigned integer less than n, return an array of length n + 1
// where array[i] := minimum index k such that get_key(k) >= i.
// Note that array[n] = the total number of keys, num_keys.
std::vector<gbbs::uintE> GetOffsets(
    const std::function<gbbs::uintE(std::size_t)>& get_key,
    gbbs::uintE num_keys, std::size_t n);

template<class Graph>
std::tuple<std::vector<double>, double, std::size_t> ComputeModularityConfig(
  Graph* graph, double resolution){
  std::vector<double> node_weights(graph->n);
  pbbs::parallel_for(0, graph->n, [&](std::size_t i){
    auto vtx = graph->get_vertex(i);
    auto wgh = vtx.getOutDegree();
    // TODO: this assumes unit edge weights
    node_weights[i] = wgh;
  });
  double total_edge_weight = parallel::ReduceAdd(absl::Span<const double>(node_weights));
  double new_resolution = resolution / total_edge_weight;
  return std::make_tuple(node_weights, new_resolution, total_edge_weight);
}


template<class Graph>
std::tuple<std::vector<double>, double, std::size_t> SeqComputeModularityConfig(
  Graph* graph, double resolution){
  std::size_t total_edge_weight = 0;
  std::vector<double> node_weights(graph->n);
  for (std::size_t i = 0; i < graph->n; i++) {
    auto vtx = graph->get_vertex(i);
    auto wgh = vtx.getOutDegree();
    // TODO: this assumes unit edge weights
    total_edge_weight += wgh;
    node_weights[i] = wgh;
  }
  double new_resolution = resolution / total_edge_weight;
  return std::make_tuple(node_weights, new_resolution, total_edge_weight);
}

// Using parallel sorting, compute inter cluster edges given a set of
// cluster_ids that form the vertices of the new graph. Uses aggregate_func
// to combine multiple edges on the same cluster ids. Returns sorted
// edges and offsets array in edges and offsets respectively.
// The number of compressed vertices should be 1 + the maximum cluster id
// in cluster_ids.
template<class Graph>
OffsetsEdges ComputeInterClusterEdgesSort(
    Graph& original_graph,
    const std::vector<gbbs::uintE>& cluster_ids,
    std::size_t num_compressed_vertices,
    const std::function<float(float, float)>& aggregate_func,
    const std::function<bool(gbbs::uintE, gbbs::uintE)>& is_valid_func) {
  // Retrieve all valid edges, mapped to cluster_ids
  auto inter_cluster_edges =
      RetrieveInterClusterEdges(original_graph, cluster_ids, is_valid_func);

  // Sort inter-cluster edges and obtain boundary indices where edges differ
  // (in any vertex). These indices are stored in filtered_mark_edges.
  auto get_endpoints =
      [](const std::tuple<gbbs::uintE, gbbs::uintE, float>& edge_with_weight) {
        return std::tie(std::get<0>(edge_with_weight),
                        std::get<1>(edge_with_weight));
      };
  auto inter_cluster_edges_sort = research_graph::parallel::ParallelSampleSort<
      std::tuple<gbbs::uintE, gbbs::uintE, float>>(
      absl::Span<std::tuple<gbbs::uintE, gbbs::uintE, float>>(
          inter_cluster_edges.data(), inter_cluster_edges.size()),
      [&](std::tuple<gbbs::uintE, gbbs::uintE, float> a,
          std::tuple<gbbs::uintE, gbbs::uintE, float> b) {
        return get_endpoints(a) < get_endpoints(b);
      });

  std::vector<gbbs::uintE> filtered_mark_edges =
      research_graph::parallel::GetBoundaryIndices<gbbs::uintE>(
          inter_cluster_edges_sort.size(),
          [&inter_cluster_edges_sort, &get_endpoints](std::size_t i,
                                                      std::size_t j) {
            return get_endpoints(inter_cluster_edges_sort[i]) ==
                   get_endpoints(inter_cluster_edges_sort[j]);
          });
  std::size_t num_filtered_mark_edges = filtered_mark_edges.size() - 1;

  // Filter out unique edges into edges
  // This is done by iterating over the boundaries where edges differ,
  // and retrieving all of the same edges in one section. These edges
  // are combined using aggregate_func.
  std::unique_ptr<std::tuple<gbbs::uintE, float>[]> edges(
      new std::tuple<gbbs::uintE, float>[num_filtered_mark_edges]);
  // Separately save the first vertex in the corresponding edges, to compute
  // offsets
  std::vector<gbbs::uintE> edges_for_offsets(num_filtered_mark_edges);
  pbbs::parallel_for(0, num_filtered_mark_edges, [&](std::size_t i) {
    // Combine edges from start_edge_index to end_edge_index
    gbbs::uintE start_edge_index = filtered_mark_edges[i];
    gbbs::uintE end_edge_index = filtered_mark_edges[i + 1];
    /*float weight = 0;
    for (std::size_t k = start_edge_index; k < end_edge_index; k++) {
      weight += std::get<2>(inter_cluster_edges_sort[k]);
    }*/
    float weight = std::get<2>(research_graph::parallel::Reduce<
                               std::tuple<gbbs::uintE, gbbs::uintE, float>>(
        absl::Span<const std::tuple<gbbs::uintE, gbbs::uintE, float>>(
            inter_cluster_edges_sort.begin() + start_edge_index,
            end_edge_index - start_edge_index),
        [&](std::tuple<gbbs::uintE, gbbs::uintE, float> a,
            std::tuple<gbbs::uintE, gbbs::uintE, float> b) {
          return std::make_tuple(
              gbbs::uintE{0}, gbbs::uintE{0},
              aggregate_func(std::get<2>(a), std::get<2>(b)));
        },
        std::make_tuple(gbbs::uintE{0}, gbbs::uintE{0}, float{0})));
    edges[i] = std::make_tuple(
        std::get<1>(inter_cluster_edges_sort[start_edge_index]), weight);
    edges_for_offsets[i] =
        std::get<0>(inter_cluster_edges_sort[start_edge_index]);
  });

  // Compute offsets using filtered edges.
  auto offsets = GetOffsets(
      [&edges_for_offsets](std::size_t i) -> gbbs::uintE {
        return edges_for_offsets[i];
      },
      num_filtered_mark_edges, num_compressed_vertices);
  return OffsetsEdges{offsets, std::move(edges), num_filtered_mark_edges};
}

// Given an array of edges (given by a tuple consisting of the second endpoint
// and a weight if the edges are weighted) and the offsets marking the index
// of the first edge corresponding to each vertex (essentially, CSR format),
// return the corresponding graph in GBBS format.
// Note that the returned graph takes ownership of the edges array.
template <typename WeightType>
std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, WeightType>>
MakeGbbsGraph(
    const std::vector<gbbs::uintE>& offsets, std::size_t num_vertices,
    std::unique_ptr<std::tuple<gbbs::uintE, WeightType>[]> edges_pointer,
    std::size_t num_edges) {
  gbbs::symmetric_vertex<WeightType>* vertices =
      new gbbs::symmetric_vertex<WeightType>[num_vertices];
  auto edges = edges_pointer.release();

  pbbs::parallel_for(0, num_vertices, [&](std::size_t i) {
    gbbs::vertex_data vertex_data{offsets[i], offsets[i + 1] - offsets[i]};
    vertices[i] = gbbs::symmetric_vertex<WeightType>(edges, vertex_data);
  });

  return std::make_unique<
      gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, WeightType>>(
      num_vertices, num_edges, vertices, [=]() {
        delete[] vertices;
        delete[] edges;
      });
}

template <typename WeightType>
std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, WeightType>>
SeqMakeGbbsGraph(
    const std::vector<gbbs::uintE>& offsets, std::size_t num_vertices,
    std::unique_ptr<std::tuple<gbbs::uintE, WeightType>[]> edges_pointer,
    std::size_t num_edges) {
  gbbs::symmetric_vertex<WeightType>* vertices =
      new gbbs::symmetric_vertex<WeightType>[num_vertices];
  auto edges = edges_pointer.release();

  for (std::size_t i = 0; i < num_vertices; i++) {
    gbbs::vertex_data vertex_data{offsets[i], offsets[i + 1] - offsets[i]};
    vertices[i] = gbbs::symmetric_vertex<WeightType>(edges, vertex_data);
  }

  return std::make_unique<
      gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, WeightType>>(
      num_vertices, num_edges, vertices, [=]() {
        delete[] vertices;
        delete[] edges;
      });
}

// Given new cluster ids in compressed_cluster_ids, remap the original
// cluster ids. A cluster id of UINT_E_MAX indicates that the vertex
// has already been placed into a finalized cluster, and this is
// preserved in the remapping.
std::vector<gbbs::uintE> FlattenClustering(
    const std::vector<gbbs::uintE>& cluster_ids,
    const std::vector<gbbs::uintE>& compressed_cluster_ids);

}  // namespace research_graph

#endif  // RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PARALLEL_GRAPH_UTILS_H_