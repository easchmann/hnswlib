#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>

namespace hnswlib {

// instrumentation for query analysis

struct QueryStats {
    // upper layer stats
    float entry_point_distance = 0.0;              // distance from query to global entry point
    float base_layer_entry_distance = 0.0;         // distance after upper layer descent
    size_t upper_layer_distance_computations = 0;
    // size_t layer_visit_counts[MAX_LAYERS] = {};
    // sized to maxlevel_ at query time
    std::vector<size_t> layer_visit_counts;         // visits per upper layer

    // base layer stats (filled in by searchBaseLayerST)
    size_t base_layer_visited_count = 0;
    size_t base_layer_distance_computations = 0;
    size_t candidates_remaining_at_termination = 0;
    std::vector<float> lowerbound_trace;            // history of lowerbound per iteration

    // all nodes whose distance was computed during base-layer search:
    // (internal_id, dist_to_query).  Includes the entry point and every
    // neighbor examined in the main loop.  Cleared on each searchKnn call.
    std::vector<std::pair<unsigned int, float>> base_layer_visited_nodes;
};
inline thread_local QueryStats last_query_stats;


typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template<typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
 public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    // added for array guard
    // static constexpr int MAX_LAYERS = 32;

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t ef_{ 0 };

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 };

    char *data_level0_memory_{nullptr};
    char **linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element

    size_t data_size_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};

    bool allow_replace_deleted_ = false;  // flag to replace deleted elements (marked as deleted) during insertions

    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements

    // adaptation for poolAndRewire: entry-point pool (node_id, last_used_query_counter)
    std::vector<std::pair<tableint, size_t>> entry_point_pool_;
    mutable std::mutex entry_pool_mutex_;
    size_t entry_pool_query_counter_{0};


    HierarchicalNSW(SpaceInterface<dist_t> *s) {
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        const std::string &location,
        bool nmslib = false,
        size_t max_elements = 0,
        bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        loadIndex(location, s, max_elements);
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        size_t max_elements,
        size_t M = 16,
        size_t ef_construction = 200,
        size_t random_seed = 100,
        bool allow_replace_deleted = false)
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
            link_list_locks_(max_elements),
            element_levels_(max_elements),
            allow_replace_deleted_(allow_replace_deleted) {
        max_elements_ = max_elements;
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        if ( M <= 10000 ) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;

        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");

        cur_element_count = 0;

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        // initializations for special treatment of the first node
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }


    ~HierarchicalNSW() {
        clear();
    }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        for (tableint i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);
    }


    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };


    void setEf(size_t ef) {
        ef_ = ef;
    }


    inline std::mutex& getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }


    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        return return_label;
    }


    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
    }


    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }


    inline char *getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }


    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) r;
    }

    size_t getMaxElements() {
        return max_elements_;
    }

    size_t getCurrentElementCount() {
        return cur_element_count;
    }

    size_t getDeletedCount() {
        return num_deleted_;
    }

    // ef_override: if > 0, use this instead of ef_construction_ (allows high-ef
    // repair searches without changing the global ef_construction_ setting).
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer, size_t ef_override = 0) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        const size_t ef = (ef_override > 0) ? ef_override : ef_construction_;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

            int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            if (layer == 0) {
                data = (int*)get_linklist0(curNodeNum);
            } else {
                if (element_levels_[curNodeNum] < layer) {
                    // node doesn't have a link list at this layer (e.g. entry point set to a
                    // low-level node); skip expanding its neighbours to avoid reading linkLists_[curNodeNum]
                    // which may be nullptr or too short.
                    continue;
                }
                data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
            }
            size_t size = getListCount((linklistsizeint*)data);
            tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);

                if (top_candidates.size() < ef || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }


    // Repair base-layer (layer-0) connections for the k_nodes nodes nearest to
    // target_data.
    //
    // Why this works where the anchor-chain approach does not:
    //   The anchor chain manipulates layer 1+ to improve navigation, but back-edges
    //   are never added to existing nodes, so greedy descent from the original entry
    //   point can never reach the new anchors.  Additionally, the entry-point
    //   heuristic used in the previous approach evaluates nodes FROM the original
    //   (far) entry point, biasing selection away from the shifted region.
    //
    //   This method bypasses both problems:
    //   1. Find k_nodes near target_data with a high-ef upper-layer descent + base
    //      layer search (no heuristic, no bias).
    //   2. For each found node, re-run searchBaseLayer starting FROM THAT NODE ITSELF
    //      with ef_repair candidates.  This is independent of the global entry point.
    //   3. Apply the MRNG heuristic to pick the best maxM0_ connections and overwrite
    //      the node's layer-0 link list.
    //
    // The entry point is NOT changed here; call setEntryPoint separately.
    // Returns the internal ids of the nodes that were repaired.
    std::vector<tableint>
    repairBaseLayer(const void* target_data, size_t k_nodes, size_t ef_repair) {
        ef_repair = std::max(ef_repair, k_nodes);

        // --- Step 1: find k_nodes nearest to target_data ---
        // Use the full upper-layer descent (same as searchKnn) to seed the
        // base-layer search as well as possible given the current graph.
        tableint seed = enterpoint_node_;
        dist_t seed_dist = fstdistfunc_(target_data, getDataByInternalId(seed), dist_func_param_);
        for (int level = maxlevel_; level > 0; level--) {
            if (level > element_levels_[seed]) continue;
            bool changed = true;
            while (changed) {
                changed = false;
                std::unique_lock<std::mutex> lock(link_list_locks_[seed]);
                unsigned int* data = get_linklist_at_level(seed, level);
                int size = getListCount(data);
                tableint* datal = (tableint*)(data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    dist_t d = fstdistfunc_(target_data, getDataByInternalId(cand), dist_func_param_);
                    if (d < seed_dist) {
                        seed_dist = d;
                        seed = cand;
                        changed = true;
                    }
                }
            }
        }

        // High-ef base-layer search from the best seeded position.
        auto pool = searchBaseLayer(seed, target_data, 0, ef_repair);

        // Extract all results; pool is a max-heap (furthest on top).
        // The k_nodes nearest are at the tail after draining.
        std::vector<std::pair<dist_t, tableint>> all;
        all.reserve(pool.size());
        while (!pool.empty()) {
            all.push_back(pool.top());
            pool.pop();
        }
        // all[0] = furthest, all.back() = nearest
        size_t start = (all.size() > k_nodes) ? all.size() - k_nodes : 0;

        std::vector<tableint> repaired;
        repaired.reserve(all.size() - start);
        for (size_t i = start; i < all.size(); i++) {
            repaired.push_back(all[i].second);
        }

        // --- Step 2: repair each found node's layer-0 connections ---
        // Search starts FROM THE NODE ITSELF — no dependence on the global
        // entry point, so the result is always correctly seeded.
        for (tableint node : repaired) {
            auto top = searchBaseLayer(node, getDataByInternalId(node), 0, ef_repair);

            // Remove self to avoid self-loops.
            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst> filtered;
            while (!top.empty()) {
                if (top.top().second != node)
                    filtered.push(top.top());
                top.pop();
            }

            // MRNG heuristic selects the best maxM0_ connections.
            getNeighborsByHeuristic2(filtered, maxM0_);

            // Overwrite the node's layer-0 link list.
            std::unique_lock<std::mutex> lock(link_list_locks_[node]);
            linklistsizeint* ll = get_linklist0(node);
            size_t n = filtered.size();
            setListCount(ll, n);
            tableint* data = (tableint*)(ll + 1);
            size_t idx = 0;
            while (!filtered.empty()) {
                data[idx++] = filtered.top().second;
                filtered.pop();
            }
        }

        return repaired;
    }


    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;
        
        std::vector<float> lowerbound_trace;
        dist_t lowerBound;
        if (bare_bone_search ||
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            last_query_stats.base_layer_visited_nodes.emplace_back(ep_id, (float)dist);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                last_query_stats.candidates_remaining_at_termination = candidate_set.size();
                break;
            }
            candidate_set.pop();

            // increments by one per node processed
            last_query_stats.base_layer_visited_count++;
            // record lowerBound in each iteration
            lowerbound_trace.push_back(lowerBound);

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    // increments by one for each call to distfunc
                    last_query_stats.base_layer_distance_computations++;
                    last_query_stats.base_layer_visited_nodes.emplace_back(candidate_id, (float)dist);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }
        
        last_query_stats.lowerbound_trace = lowerbound_trace;

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }


    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                        getDataByInternalId(curent_pair.second),
                                        dist_func_param_);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }


    linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }


    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }


    tableint mutuallyConnectNewElement(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_);
        if (top_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint *data = (tableint *) (ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_), data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
        }

        return next_closest_entry_point;
    }


    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);

        std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        // Reallocate all other layers
        char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        linkLists_ = linkLists_new;

        max_elements_ = new_max_elements;
    }

    size_t indexFileSize() const {
        size_t size = 0;
        size += sizeof(offsetLevel0_);
        size += sizeof(max_elements_);
        size += sizeof(cur_element_count);
        size += sizeof(size_data_per_element_);
        size += sizeof(label_offset_);
        size += sizeof(offsetData_);
        size += sizeof(maxlevel_);
        size += sizeof(enterpoint_node_);
        size += sizeof(maxM_);

        size += sizeof(maxM0_);
        size += sizeof(M_);
        size += sizeof(mult_);
        size += sizeof(ef_construction_);

        size += cur_element_count * size_data_per_element_;

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }
        return size;
    }

    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);

        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                output.write(linkLists_[i], linkListSize);
        }
        output.close();
    }


    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        clear();
        // get file size:
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = input.tellg();

        /// Optional - check if index is ok:
        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count; i++) {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                throw std::runtime_error("Index seems to be corrupted or unsupported");
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize != 0) {
                input.seekg(linkListSize, input.cur);
            }
        }

        // throw exception if it either corrupted or old index
        if (input.tellg() != total_filesize)
            throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();
        /// Optional check end

        input.seekg(pos, input.beg);

        data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *) malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                input.read(linkLists_[i], linkListSize);
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i);
            }
        }

        input.close();

        return;
    }


    template<typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        
        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        char* data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *) dist_func_param_);
        std::vector<data_t> data;
        data_t* data_ptr = (data_t*) data_ptrv;
        for (size_t i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }


    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    void markDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        markDeletedInternal(internalId);
    }


    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    void markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }


    /*
    * Removes the deleted mark of the node, does NOT really change the current graph.
    * 
    * Note: the method is not safe to use when replacement of deleted elements is enabled,
    *  because elements marked as deleted can be completely removed by addPoint
    */
    void unmarkDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }



    /*
    * Remove the deleted mark of the node.
    */
    void unmarkDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error("The requested to undelete element is not deleted");
        }
    }


    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    bool isMarkedDeleted(tableint internalId) const {
        unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }


    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }


    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }


    /*
    * Adds point. Updates the point if it is already in the index.
    * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
    */
    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addPoint(data_point, label, -1);
            return;
        }
        // check if there is vacant place
        tableint internal_id_replaced;
        std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) {
            internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();

        // if there is no vacant place then add or update point
        // else add point to vacant place
        if (!is_vacant_place) {
            addPoint(data_point, label, -1);
        } else {
            // we assume that there are no concurrent operations on deleted element
            labeltype label_replaced = getExternalLabel(internal_id_replaced);
            setExternalLabel(internal_id_replaced, label);

            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }


    void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
        // update the feature vector associated with existing point with new vector
        memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

        int maxLevelCopy = maxlevel_;
        tableint entryPointCopy = enterpoint_node_;
        // If point to be updated is entry point and graph just contains single element then just return.
        if (entryPointCopy == internalId && cur_element_count == 1)
            return;

        int elemLevel = element_levels_[internalId];
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<tableint> sCand;
            std::unordered_set<tableint> sNeigh;
            std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
            if (listOneHop.size() == 0)
                continue;

            sCand.insert(internalId);

            for (auto&& elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(update_probability_generator_) > updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);

                std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                for (auto&& elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto&& neigh : sNeigh) {
                // if (neigh == internalId)
                //     continue;

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;  // sCand guaranteed to have size >= 1
                size_t elementsToKeep = std::min(ef_construction_, size);
                for (auto&& cand : sCand) {
                    if (cand == neigh)
                        continue;

                    dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                    if (candidates.size() < elementsToKeep) {
                        candidates.emplace(distance, cand);
                    } else {
                        if (distance < candidates.top().first) {
                            candidates.pop();
                            candidates.emplace(distance, cand);
                        }
                    }
                }

                // Retrieve neighbours using heuristic and set connections.
                getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                {
                    std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                    linklistsizeint *ll_cur;
                    ll_cur = get_linklist_at_level(neigh, layer);
                    size_t candSize = candidates.size();
                    setListCount(ll_cur, candSize);
                    tableint *data = (tableint *) (ll_cur + 1);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        data[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
    }


    void repairConnectionsForUpdate(
        const void *dataPoint,
        tableint entryPointInternalId,
        tableint dataPointInternalId,
        int dataPointLevel,
        int maxLevel) {
        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                    for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                        tableint cand = datal[i];
                        dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        if (dataPointLevel > maxLevel)
            throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                    currObj, dataPoint, level);

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
            while (topCandidates.size() > 0) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }

            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
            // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
            if (filteredTopCandidates.size() > 0) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
            }
        }
    }


    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
        std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
        unsigned int *data = get_linklist_at_level(internalId, level);
        int size = getListCount(data);
        std::vector<tableint> result(size);
        tableint *ll = (tableint *) (data + 1);
        memcpy(result.data(), ll, size * sizeof(tableint));
        return result;
    }


    tableint addPoint(const void *data_point, labeltype label, int level) {
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    }
                }
                lock_table.unlock();

                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                updatePoint(data_point, existingInternalId, 1.0);

                return existingInternalId;
            }

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel;

        std::unique_lock <std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;

        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

        // Initialisation of the data and label
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);

        if (curlevel) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    // guard against an entry point that was set below maxlevelcopy
                    if (level > element_levels_[currObj]) continue;
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *) (data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)  // possible?
                    throw std::runtime_error("Level error");

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                if (epDeleted) {
                    top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            }
        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }


    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        // reset query stats
        last_query_stats = QueryStats{};

        // entry point distance: how far is the fixed entry point from the query?
        // if this grows after a distribution shift, the entry point might be a problem
        last_query_stats.entry_point_distance = curdist;

        last_query_stats.layer_visit_counts.assign(maxlevel_ + 1, 0);


        for (int level = maxlevel_; level > 0; level--) {
            // currObj may be below this level (e.g. after set_entry_point with a low-level node).
            // reading get_linklist(currObj, level) in that case reads past the node's allocated link list buffer and produces garbage cand values -> skip the level instead.
            if (level > element_levels_[currObj]) {
                last_query_stats.layer_visit_counts[level] = 0;
                continue;
            }
            bool changed = true;

            // upper layer visit count per layer
            size_t layer_visits = 0;

            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    // count distance computations (increments with each call of distfunc)
                    last_query_stats.upper_layer_distance_computations++;

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }

                    // count every neighbor examined
                    layer_visits++;
                }
            }

            // store visits for this layer (layer index 1 to maxlevel_)
            // if (level < MAX_LAYERS) { 
            //     last_query_stats.layer_visit_counts[level] = layer_visits;
            // }
            last_query_stats.layer_visit_counts[level] = layer_visits;

        }
        // after the upper layer for loop finishes, curdist holds the distance to the best entry node found by the descent
        last_query_stats.base_layer_entry_distance = curdist;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerST<true>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerST<false>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }


    std::vector<std::pair<dist_t, labeltype >>
    searchStopConditionClosest(
        const void *query_data,
        BaseSearchStopCondition<dist_t>& stop_condition,
        BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::vector<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            if (level > element_levels_[currObj]) continue;
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

        size_t sz = top_candidates.size();
        result.resize(sz);
        while (!top_candidates.empty()) {
            result[--sz] = top_candidates.top();
            top_candidates.pop();
        }

        stop_condition.filter_results(result);

        return result;
    }


    void checkIntegrity() {
        int connections_checked = 0;
        std::vector <int > inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);
                std::unordered_set<tableint> s;
                for (int j = 0; j < size; j++) {
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                assert(s.size() == size);
            }
        }
        if (cur_element_count > 1) {
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i=0; i < cur_element_count; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        }
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    }

    // set global entry point to a specific node (already existing in the index)
    void setEntryPoint(tableint new_entry_point){
        if (new_entry_point >= cur_element_count){
            throw std::runtime_error("setEntryPoint: node ID out of range");
        }
        std::unique_lock<std::mutex> lock(global);
        enterpoint_node_ = new_entry_point;

        // maxlevel_ always reflects the true graph maximum (not clamped to the entry point's level).
        // searchKnn and addPoint guard against accessing levels above element_levels_[currObj].
        int node_level = element_levels_[new_entry_point];
        if (node_level > maxlevel_) {
            maxlevel_ = node_level;
        }
    }

    // returns the internal ids of all nodes that exist at >=min_layer
    //note: internal ids != external labels -> can be converted using getExternalLabel()
    std::vector<tableint> getNodesAtLayer(int min_layer) const{
        std::vector<tableint> result;
        // reserve memor to avoid reallocation, rough estimate of 10% nodes per layer
        result.reserve(cur_element_count/10);

        for (tableint i=0; i<cur_element_count; i++){
            if (element_levels_[i]>=min_layer){
                result.push_back(i);
            }
        }
        return result;

    }

    // inject a directed edge src->dst into layer-0; no-op if dst already a neighbor or src is at capacity
    void add_layer0_edge(tableint src, tableint dst) {
        linklistsizeint *ll = get_linklist0(src);
        size_t sz = getListCount(ll);
        if (sz >= maxM0_) return;
        tableint *data = (tableint *)(ll + 1);
        for (size_t i = 0; i < sz; i++)
            if (data[i] == dst) return;
        data[sz] = dst;
        setListCount(ll, sz + 1);
    }

    // inject src->dst into layer-0 with eviction: if at capacity, evict the farthest neighbor if dst is closer
    bool add_layer0_edge_evict(tableint src, tableint dst) {
        if (src == dst) return false;
        linklistsizeint *ll = get_linklist0(src);
        size_t sz = getListCount(ll);
        tableint *data = (tableint *)(ll + 1);

        for (size_t i = 0; i < sz; i++)
            if (data[i] == dst) return false;

        if (sz < maxM0_) {
            data[sz] = dst;
            setListCount(ll, sz + 1);
            return true;
        }

        const void *src_vec = getDataByInternalId(src);
        dist_t dst_dist = fstdistfunc_(src_vec, getDataByInternalId(dst), dist_func_param_);

        size_t farthest_idx = 0;
        dist_t farthest_dist = 0;
        for (size_t i = 0; i < sz; i++) {
            dist_t d = fstdistfunc_(src_vec, getDataByInternalId(data[i]), dist_func_param_);
            if (d > farthest_dist) { farthest_dist = d; farthest_idx = i; }
        }

        if (dst_dist < farthest_dist) {
            data[farthest_idx] = dst;
            return true;
        }
        return false;
    }

    // promote an existing node to target_level and wire it into the graph at every new level/layer using the standard neighbour selection heuristic
    // only difference to addPoint is that the node already exisits in the index and we only add upper-layer connections for a node that previously existed at lower levels only.
    tableint promoteNodeToLayer(tableint node_id, int target_level){
        if (node_id >= cur_element_count){
            throw std::runtime_error("promoteNodeToLayer: node ID out of range");
        }
        if (target_level < 1){
            throw std::runtime_error("promoteNodeToLayer: target layer must be >=1");
        }

        int curr_level = element_levels_[node_id];
        if (curr_level >= target_level){
            //node already at desired layer, no-op
            return node_id;
        }

        //allocate a bigger link list block to cover the new levels
        // layout: $target_level chunks of size_links_per_element (one per upper layer)
        char *new_linklist = (char *)malloc(size_links_per_element_ * target_level + 1);
        if (new_linklist == nullptr){
            throw std::runtime_error("promoteNodeToLayer: malloc failed");
        }
        memset(new_linklist, 0, size_links_per_element_ * target_level +1);

        // copy all existing links to new block
        if (curr_level > 0 && linkLists_[node_id]!=nullptr){
            memcpy(new_linklist, linkLists_[node_id], size_links_per_element_ * curr_level);
            free(linkLists_[node_id]);
    
        }
        linkLists_[node_id] = new_linklist;
        element_levels_[node_id] = target_level;

        //descend from the current entry point to target_level + 1 to find good node for entry node for the target level
        tableint curr_obj = enterpoint_node_;
        dist_t curr_dist = fstdistfunc_(getDataByInternalId(node_id), getDataByInternalId(curr_obj), dist_func_param_);

        for (int level= maxlevel_; level>target_level; level--){
            if (level > element_levels_[curr_obj]) continue;
            // greedy best-first search at a single layer.
            // keeps running as long as it finds a better node.  If any neighbour turns out to be closer to node_id than the current best, changed is set back to true and the loop runs again from that new node.
            // after a full iteration through all neighbours finds no better node, loop exits (local minimum -> closest reachable node at layer given the graph structure)
            bool changed = true;
            while (changed){
                changed = false;
                std::unique_lock<std::mutex> lock(link_list_locks_[curr_obj]);
                // array stored as [count | neighbour0 | neighbour1 | ...]
                int *data = (int *)get_linklist(curr_obj, level);
                // reads the count from the first entry
                int size = getListCount((linklistsizeint *)data);
                // data + 1 steps past count entry to where the actual neighbour ids start
                tableint *neighbours = (tableint *)(data + 1);
                for (int i=0; i < size; i++){
                    tableint candidate = neighbours[i];
                    dist_t dist = fstdistfunc_(getDataByInternalId(node_id), getDataByInternalId(candidate), dist_func_param_);
                    if (dist < curr_dist){
                        curr_dist = dist;
                        curr_obj= candidate;
                        changed = true;
                    }
                }
            }
        }
        // wire the node into each new level top down (same as in addPoint)
        for (int level = std::min(target_level, maxlevel_); level > curr_level; level--){
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(curr_obj, getDataByInternalId(node_id), level);
            
            // filter self from candidates to avoid self-loops
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filtered;
            while (!top_candidates.empty()) {
                if (top_candidates.top().second != node_id){
                    filtered.push(top_candidates.top());
                }
                top_candidates.pop();
            }

        // removed since produced errors
        // if (!filtered.empty()){
        //     curr_obj = mutuallyConnectNewElement(getDataByInternalId(node_id), node_id, filtered, level, true);
        // }

        // apply neighbour heuristic to select final M neighbours
        getNeighborsByHeuristic2(filtered, M_);

        // write forward edges, no back-edges to avoid corrupting existing nodes' link lists at a layer they were not built for
        std::unique_lock<std::mutex> lock(link_list_locks_[node_id]);
        linklistsizeint *ll = get_linklist(node_id, level);
        size_t n = filtered.size();
        setListCount(ll, n);
        tableint *data = (tableint *)(ll + 1);
        size_t idx = 0;
        while (!filtered.empty()) {
            data[idx++] = filtered.top().second;
            filtered.pop();
        }

        // update curr_obj for the next layer down
        if (n > 0) curr_obj = data[0];

        }
        //update global entry point if it reaches new max_layer
        // added outer brackets to determine scope and thus lifetime of the lock
        {   
            std::unique_lock<std::mutex> lock(global);
            if (target_level > maxlevel_){
                maxlevel_ = target_level;
                enterpoint_node_ = node_id;
            }
        }

        return node_id;

    }

    // add edges from existing nodes to a target node
    void addDirectedEdges(const void*target_data, int layer, size_t k_nodes){
        if (layer < 1 || layer > maxlevel_){
            throw std::runtime_error("addDirectedEdges: layer is invalid");
        }

        //descend to the target layer
        tableint curr_obj = enterpoint_node_;
        dist_t curr_dist = fstdistfunc_(target_data, getDataByInternalId(curr_obj), dist_func_param_);

        for (int level= maxlevel_; level>layer; level--){
            if (level > element_levels_[curr_obj]) continue;
            bool changed = true;
            while (changed){
                changed = false;
                std::unique_lock<std::mutex> lock(link_list_locks_[curr_obj]);
                // array stored as [count | neighbour0 | neighbour1 | ...]
                int *data = (int *)get_linklist(curr_obj, level);
                // reads the count from the first entry
                int size = getListCount((linklistsizeint *)data);
                // data + 1 steps past count entry to where the actual neighbour ids start
                tableint *neighbours = (tableint *)(data + 1);
                for (int i=0; i < size; i++){
                    tableint candidate = neighbours[i];
                    dist_t dist = fstdistfunc_(target_data, getDataByInternalId(candidate), dist_func_param_);
                    if (dist < curr_dist){
                        curr_dist = dist;
                        curr_obj= candidate;
                        changed = true;
                    }
                }
            }
        }
        // find k_nodes nearest nodes to target at this layer
        // searchBaseLayer returns a max-heap (priority queue where the furthest element is at the top)
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,  CompareByFirst> candidates = searchBaseLayer(curr_obj, target_data, layer);

        std::vector<tableint> nodes_to_update;
        nodes_to_update.reserve(k_nodes);

        // candidates.top() returns the pair with the largest distance in the heap
        std::vector<std::pair<dist_t, tableint>> all;
        while (!candidates.empty()) {
                all.push_back(candidates.top());
                candidates.pop();   
        }
        // last k_nodes elements are the nearest (smallest distance)
        for (size_t i = std::max((size_t)0, all.size() - k_nodes); i < all.size(); i++){
            nodes_to_update.push_back(all[i].second);
        }


        // for each selected node rerun neighbour selectio using target_data as the search centre for the heuristic to form an edge in that direction
        //
        for (tableint node:nodes_to_update){
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(node, target_data, layer);

            // fixed: filter self to avoid "Trying to connect an element to itself"
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filtered;
            while (!top_candidates.empty()) {
                if (top_candidates.top().second != node)
                    filtered.push(top_candidates.top());
                top_candidates.pop();
            }

            if (!filtered.empty())
                mutuallyConnectNewElement(getDataByInternalId(node), node, filtered, layer, true);
        }
    }

    // Add dst to src's neighbor list at `level`, evicting the farthest current neighbor if the list is already at capacity. 
    // No-op if dst is already present.  Both src and dst must exist at `level`.
    bool addBackEdge(tableint src, tableint dst, int level) {
        if (src >= cur_element_count)
            throw std::runtime_error("addBackEdge: src out of range");
        if (dst >= cur_element_count)
            throw std::runtime_error("addBackEdge: dst out of range");
        if (src == dst) return false;
        if (level < 0 || level > maxlevel_)
            throw std::runtime_error("addBackEdge: level out of range");
        if (level > element_levels_[src])
            throw std::runtime_error("addBackEdge: src does not exist at level");
        if (level > element_levels_[dst])
            throw std::runtime_error("addBackEdge: dst does not exist at level");

        size_t max_size = (level == 0) ? maxM0_ : maxM_;

        std::unique_lock<std::mutex> lock(link_list_locks_[src]);
        linklistsizeint* ll = (level == 0) ? get_linklist0(src) : (linklistsizeint*)get_linklist(src, level);
        size_t cur_size = getListCount(ll);
        tableint* neighbors = (tableint*)(ll + 1);

        // Idempotent: skip if dst is already a neighbor of src
        for (size_t i = 0; i < cur_size; i++) {
            if (neighbors[i] == dst) return false;
        }

        if (cur_size < max_size) {
            // There is room —> just append
            neighbors[cur_size] = dst;
            setListCount(ll, cur_size + 1);
            return true;
        } else {
            // List full —> evict the farthest current neighbor if dst is closer
            const void* src_data = getDataByInternalId(src);
            dist_t dst_dist = fstdistfunc_(getDataByInternalId(dst), src_data, dist_func_param_);

            size_t worst_idx = 0;
            dist_t worst_dist = fstdistfunc_(getDataByInternalId(neighbors[0]), src_data, dist_func_param_);
            for (size_t i = 1; i < cur_size; i++) {
                dist_t d = fstdistfunc_(getDataByInternalId(neighbors[i]), src_data, dist_func_param_);
                if (d > worst_dist) { worst_dist = d; worst_idx = i; }
            }
            if (dst_dist < worst_dist) {
                neighbors[worst_idx] = dst;
                return true;
            }
            return false;
        }
    }

    void rewireLocalNeighbourhood(const void *target_data, int layer, size_t k_nodes){
        if (layer < 0 || layer > maxlevel_){
            throw std::runtime_error("rewireLocalNeighbourhood: layer is invalid");
        }

        //descend to the target layer
        tableint curr_obj = enterpoint_node_;
        dist_t curr_dist = fstdistfunc_(target_data, getDataByInternalId(curr_obj), dist_func_param_);

        for (int level= maxlevel_; level>layer; level--){
            if (level > element_levels_[curr_obj]) continue;
            bool changed = true;
            while (changed){
                changed = false;
                std::unique_lock<std::mutex> lock(link_list_locks_[curr_obj]);
                // array stored as [count | neighbour0 | neighbour1 | ...]
                int *data = (int *)get_linklist(curr_obj, level);
                // reads the count from the first entry
                int size = getListCount((linklistsizeint *)data);
                // data + 1 steps past count entry to where the actual neighbour ids start
                tableint *neighbours = (tableint *)(data + 1);
                for (int i=0; i < size; i++){
                    tableint candidate = neighbours[i];
                    dist_t dist = fstdistfunc_(target_data, getDataByInternalId(candidate), dist_func_param_);
                    if (dist < curr_dist){
                        curr_dist = dist;
                        curr_obj= candidate;
                        changed = true;
                    }
                }
            }
        }

        // find the k_nodes closest nodes at specified layer
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,  CompareByFirst> candidates = searchBaseLayer(curr_obj, target_data, layer);

        std::vector<tableint> nodes_to_rewire;
        nodes_to_rewire.reserve(k_nodes);

        // candidates.top() returns the pair with the largest distance in the heap
        std::vector<std::pair<dist_t, tableint>> all;
        while (!candidates.empty()) {
            all.push_back(candidates.top());
            candidates.pop();
        }
        // last k_nodes elements are the nearest (smallest distance)
        for (size_t i = std::max((size_t)0, all.size() - k_nodes); i < all.size(); i++){
            nodes_to_rewire.push_back(all[i].second);
        }

         // recompute each node's neighbours using that node as the search centre
        // local re-optimisation rather than adding edges toward the target. resulting connections should be optimal for the node
        for (tableint node : nodes_to_rewire) {
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates =
                searchBaseLayer(node, getDataByInternalId(node), layer);

            // remove self from candidates to avoid self-loops
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filtered;
            while (!top_candidates.empty()) {
                if (top_candidates.top().second != node)
                    filtered.push(top_candidates.top());
                top_candidates.pop();
            }

            if (!filtered.empty()){
                mutuallyConnectNewElement(getDataByInternalId(node), node, filtered, layer, true);
            }
        }


    }


    // adaptation for poolAndRewire: query-driven upper-layer rewiring
    //
    // For each level from min(max_layer, maxlevel_) down to 1:
    //   1. Run greedy descent at that level:find curr_obj where descent stalls
    //   2. Run a beam search (searchBaseLayer at the same level) from curr_obj to find the globally best reachable node 
    //      for the query at that level.
    //   3. If curr_obj's distance to query > alpha * best node's distance,
    //      descent was suboptimal: add a directed edge curr_obj -> best_node,
    //      evicting the weakest (furthest-from-curr_obj) neighbor when full
    //
    // searchBaseLayer accepts a layer parameter and works correctly at level >= 1
    // (uses get_linklist(node, layer) for neighbor access at upper layers)
    //
    // Returns the number of edges added.
    int rewireForQuery(const void *query_data, int max_layer, float alpha = 1.5f){
        if (cur_element_count == 0 || max_layer < 1)
            return 0;

        int edges_added = 0;
        tableint curr_obj = enterpoint_node_;
        dist_t curr_dist = fstdistfunc_(query_data, getDataByInternalId(curr_obj), dist_func_param_);

        for (int level = std::min(max_layer, maxlevel_); level >= 1; level--){
            // greedy descent at this level
            bool changed = true;
            while (changed) {
                changed = false;
                if (element_levels_[curr_obj] < level) break;
                std::unique_lock<std::mutex> lock(link_list_locks_[curr_obj]);
                int *data = (int *)get_linklist(curr_obj, level);
                int  sz   = getListCount((linklistsizeint *)data);
                tableint *nbrs = (tableint *)(data + 1);
                for (int i = 0; i < sz; i++){
                    tableint cand = nbrs[i];
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
                    if (d < curr_dist){
                        curr_dist = d;
                        curr_obj  = cand;
                        changed   = true;
                    }
                }
            }

            // beam search at this level to find the best reachable node
            auto beam = searchBaseLayer(curr_obj, query_data, level);
            if (beam.empty()) continue;

            tableint best_node = curr_obj;
            dist_t   best_dist = curr_dist;
            while (!beam.empty()){
                if (beam.top().first < best_dist){
                    best_dist = beam.top().first;
                    best_node = beam.top().second;
                }
                beam.pop();
            }

            // if descent is suboptimal by more than alpha, add a corrective edge
            if (curr_dist > alpha * best_dist && best_node != curr_obj && element_levels_[best_node] >= level){

                std::unique_lock<std::mutex> lock(link_list_locks_[curr_obj]);
                linklistsizeint *ll = get_linklist(curr_obj, level);
                size_t cur_sz = getListCount(ll);
                tableint *ndata = (tableint *)(ll + 1);

                bool exists = false;
                for (size_t i = 0; i < cur_sz; i++){
                    if (ndata[i] == best_node){
                        exists = true; break;
                    }
                }

                if (!exists){
                    if (cur_sz < (size_t)maxM_){
                        ndata[cur_sz] = best_node;
                        setListCount(ll, cur_sz + 1);
                        edges_added++;
                    }
                    else{
                        // evict the weakest (furthest-from-curr_obj) neighbor
                        int wi = -1;
                        dist_t wd = 0;
                        for (size_t i = 0; i < cur_sz; i++){
                            dist_t d = fstdistfunc_(getDataByInternalId(curr_obj), getDataByInternalId(ndata[i]), dist_func_param_);
                            if (d > wd){
                                wd = d; wi = (int)i;
                            }
                        }
                        dist_t nd = fstdistfunc_(getDataByInternalId(curr_obj), getDataByInternalId(best_node), dist_func_param_);
                        if (wi >= 0 && nd < wd){
                            ndata[wi] = best_node;
                            edges_added++;
                        }
                    }
                }
            }

            // advance curr_obj toward query for the next layer down
            if (best_node != curr_obj && element_levels_[best_node] >= level - 1){
                curr_obj  = best_node;
                curr_dist = best_dist;
            }
        }

        return edges_added;
    }

    // adaptation for poolAndRewire:
    // Entry-point pool: small bounded set of candidate entry points covering different query regions. 
    // At query time Python calls getBestEntryPoint to get the pool member closest to the query, then calls set_entry_point on it
    // before running knn_query.

    void addToEntryPool(tableint node_id){
        if (node_id >= cur_element_count){
            throw std::runtime_error("addToEntryPool: node_id out of range");
        }
        std::unique_lock<std::mutex> lock(entry_pool_mutex_);
        for (auto &p : entry_point_pool_){
            // already present
            if (p.first == node_id) return;
        }
        entry_point_pool_.emplace_back(node_id, entry_pool_query_counter_);
    }

    // Return the pool member closest to query_data.
    // Falls back to the global entry point when the pool is empty.
    // Updates LRU timestamp for the winner after the scan (not during, to avoid aliasing when best changes mid-loop)
    tableint getBestEntryPoint(const void *query_data){
        std::unique_lock<std::mutex> lock(entry_pool_mutex_);
        ++entry_pool_query_counter_;

        if (entry_point_pool_.empty())
            return enterpoint_node_;

        tableint best = entry_point_pool_[0].first;
        dist_t best_d = fstdistfunc_(query_data, getDataByInternalId(best), dist_func_param_);
        for (auto &p : entry_point_pool_){
            dist_t d = fstdistfunc_(query_data, getDataByInternalId(p.first), dist_func_param_);
            if (d < best_d){
                best_d = d; best = p.first;
            }
        }
        // update LRU timestamp only after finding the final winner
        for (auto &p : entry_point_pool_) {
            if (p.first == best){
                p.second = entry_pool_query_counter_;
                break;
            }
        }
        return best;
    }

    // Remove LRU pool members until size <= max_size.
    void pruneEntryPool(size_t max_size){
        std::unique_lock<std::mutex> lock(entry_pool_mutex_);

        if (entry_point_pool_.size() <= max_size){
            return;
        }
        std::sort(entry_point_pool_.begin(), entry_point_pool_.end(),
                [](const auto& a, const auto& b){
                    return a.second < b.second;
                });

        entry_point_pool_.erase(entry_point_pool_.begin(), entry_point_pool_.end() - max_size);
    }

    size_t entryPoolSize() const {
        std::unique_lock<std::mutex> lock(entry_pool_mutex_);
        return entry_point_pool_.size();
    }

};



}  // namespace hnswlib
