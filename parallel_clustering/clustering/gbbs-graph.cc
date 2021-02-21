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

#include "clustering/gbbs-graph.h"

#include <algorithm>
#include <memory>

#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "clustering/status_macros.h"
#include "external/gbbs/gbbs/macros.h"

namespace research_graph {
namespace in_memory {

// TODO(jeshi,laxmand): should adjacency_list be a const&?
absl::Status GbbsGraph::Import(AdjacencyList adjacency_list) {
  return absl::OkStatus();
}

absl::Status GbbsGraph::FinishImport() {
  return absl::OkStatus();
}

gbbs::symmetric_graph<gbbs::csv_bytepd_amortized, pbbslib::empty>* GbbsGraph::Graph()
    const {
  return graph_.get();
}


}  // namespace in_memory
}  // namespace research_graph
