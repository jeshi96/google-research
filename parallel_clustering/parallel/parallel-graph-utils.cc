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

#include "parallel/parallel-graph-utils.h"

#include <cstdio>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "external/gbbs/gbbs/gbbs.h"
#include "external/gbbs/gbbs/graph_io.h"
#include "external/gbbs/gbbs/macros.h"
#include "external/gbbs/pbbslib/get_time.h"
#include "external/gbbs/pbbslib/seq.h"
#include "external/gbbs/pbbslib/sequence_ops.h"
#include "external/gbbs/pbbslib/utilities.h"
#include "parallel/parallel-sequence-ops.h"

namespace research_graph {

float FloatFromWeightPGU(float weight) { return weight; }
float FloatFromWeightPGU(pbbslib::empty weight) { return 1; }

std::vector<gbbs::uintE> GetOffsets(
    const std::function<gbbs::uintE(std::size_t)>& get_key,
    gbbs::uintE num_keys, std::size_t n) {
  std::vector<gbbs::uintE> offsets(n + 1, 0);
  // Obtain the boundary indices where keys differ
  // These indices are stored in filtered_mark_keys
  std::vector<gbbs::uintE> filtered_mark_keys =
      parallel::GetBoundaryIndices<gbbs::uintE>(
          num_keys, [&get_key](std::size_t i, std::size_t j) {
            return get_key(i) == get_key(j);
          });
  std::size_t num_filtered_mark_keys = filtered_mark_keys.size() - 1;

  // We must do an extra step for keys i which do not appear in get_key
  // At the start of each boundary index start_index, the first key
  // is given by get_key(start_index). The offset for that key is precisely
  // start_index. The offset for each key after get_key(start_index - 1) to
  // get_key(start_index) is also start_index, because these keys do
  // not appear in get_key.
  pbbs::parallel_for(0, num_filtered_mark_keys + 1, [&](std::size_t i) {
    auto start_index = filtered_mark_keys[i];
    gbbs::uintE curr_key = start_index == num_keys ? n : get_key(start_index);
    gbbs::uintE prev_key = start_index == 0 ? 0 : get_key(start_index - 1) + 1;
    for (std::size_t j = prev_key; j <= curr_key; j++) {
      offsets[j] = start_index;
    }
  });

  return offsets;
}

std::vector<gbbs::uintE> FlattenClustering(
    const std::vector<gbbs::uintE>& cluster_ids,
    const std::vector<gbbs::uintE>& compressed_cluster_ids) {
  std::vector<gbbs::uintE> new_cluster_ids(cluster_ids.size());
  pbbs::parallel_for(0, cluster_ids.size(), [&](std::size_t i) {
    new_cluster_ids[i] = (cluster_ids[i] == UINT_E_MAX)
                             ? UINT_E_MAX
                             : compressed_cluster_ids[cluster_ids[i]];
  });
  return new_cluster_ids;
}

}  // namespace research_graph