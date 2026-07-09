#pragma once

// Secondary edge store augmenting hnswlib without modifying it.
// Standalone port of src/drift/conjugate_graph.py — does not depend on HierarchicalNSW or any Index type; takes base data + dim as plain arguments.

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <algorithm>

namespace hnswlib {

struct ConjEdge {
    uint32_t neighbor_id;
    float distance;
    float t_added;
    int epoch_added;
    int traversal_count = 0;
};

struct ConjugateGraphStats {
    size_t n_nodes_with_edges;
    size_t total_edges;
    double mean_edges_per_node;
    size_t max_edges_per_node;
    double mean_traversal_count;
    double mean_staleness;
};

class ConjugateGraph {
 public:
    ConjugateGraph(size_t M_conj = 8, size_t max_total_edges = 500000)
        : M_conj_(M_conj), max_total_edges_(max_total_edges), total_edges_(0) {}

    // staleness: higher = more stale = eviction candidate
    static float staleness(const ConjEdge &e, int current_epoch, float alpha = 1.0f, float beta = 0.5f) {
        return alpha * (current_epoch - e.epoch_added) - beta * e.traversal_count;
    }

    // add directed edge src -> dst; returns true if the edge was added
    bool add_edge(uint32_t src, uint32_t dst, float distance, float t_added, int epoch_added, int current_epoch) {
        auto it = edges_.find(src);

        if (it == edges_.end()) {
            if (total_edges_ >= max_total_edges_) {
                int evicted = evict(current_epoch);
                if (evicted == 0 && total_edges_ >= max_total_edges_) return false;
            }
            edges_[src] = {ConjEdge{dst, distance, t_added, epoch_added, 0}};
            total_edges_++;
            return true;
        }

        std::vector<ConjEdge> &neighbors = it->second;

        if (neighbors.size() < M_conj_) {
            if (total_edges_ >= max_total_edges_) {
                int evicted = evict(current_epoch);
                if (evicted == 0 && total_edges_ >= max_total_edges_) return false;
            }
            neighbors.push_back(ConjEdge{dst, distance, t_added, epoch_added, 0});
            total_edges_++;
            return true;
        }

        // Node is full — evict the oldest edge if the new edge is from a later epoch.
        // Age-only: traversal_count is excluded to prevent heavily-used pre-drift edges from becoming immune to eviction during distribution shift.
        size_t oldest_idx = 0;
        for (size_t i = 1; i < neighbors.size(); i++)
            if (neighbors[i].epoch_added < neighbors[oldest_idx].epoch_added) oldest_idx = i;

        if (epoch_added > neighbors[oldest_idx].epoch_added) {
            neighbors[oldest_idx] = ConjEdge{dst, distance, t_added, epoch_added, 0};
            return true;
        }

        return false;
    }

    // return edges from node_id, incrementing traversal counts
    const std::vector<ConjEdge> &lookup(uint32_t node_id, int current_epoch) {
        (void)current_epoch;
        auto it = edges_.find(node_id);
        if (it == edges_.end()) return empty_;
        for (auto &e : it->second) e.traversal_count++;
        return it->second;
    }

    // augment HNSW top-k with conjugate graph expansion
    std::pair<std::vector<uint32_t>, std::vector<float>> enhanced_search(
            const std::vector<uint32_t> &seed_ids,
            const std::vector<float> &seed_dists,
            const float *base, size_t dim,
            const float *query_vec, size_t k, int current_epoch, bool two_hop = true) {
        std::vector<uint32_t> candidate_ids(seed_ids);
        std::vector<float> candidate_dists(seed_dists);
        std::unordered_map<uint32_t, char> seen;
        for (auto id : seed_ids) seen[id] = 1;

        // Hop 1: expand from primary HNSW results; increment traversal counts
        std::vector<uint32_t> hop1_new;
        for (auto node_id : seed_ids) {
            for (const auto &edge : lookup(node_id, current_epoch)) {
                uint32_t n = edge.neighbor_id;
                if (seen.find(n) == seen.end()) {
                    seen[n] = 1;
                    float dist = squared_l2(query_vec, base + (size_t)n * dim, dim);
                    candidate_ids.push_back(n);
                    candidate_dists.push_back(dist);
                    hop1_new.push_back(n);
                }
            }
        }

        // Hop 2: expand from hop-1 discoveries; do NOT increment traversal counts
        if (two_hop) {
            for (auto node_id : hop1_new) {
                auto it = edges_.find(node_id);
                if (it == edges_.end()) continue;
                for (const auto &edge : it->second) {
                    uint32_t n = edge.neighbor_id;
                    if (seen.find(n) == seen.end()) {
                        seen[n] = 1;
                        float dist = squared_l2(query_vec, base + (size_t)n * dim, dim);
                        candidate_ids.push_back(n);
                        candidate_dists.push_back(dist);
                    }
                }
            }
        }

        std::vector<size_t> order(candidate_ids.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = i;

        if (candidate_ids.size() <= k) {
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                return candidate_dists[a] < candidate_dists[b];
            });
        } else {
            std::nth_element(order.begin(), order.begin() + k, order.end(), [&](size_t a, size_t b) {
                return candidate_dists[a] < candidate_dists[b];
            });
            order.resize(k);
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                return candidate_dists[a] < candidate_dists[b];
            });
        }

        std::vector<uint32_t> out_ids(order.size());
        std::vector<float> out_dists(order.size());
        for (size_t i = 0; i < order.size(); i++) {
            out_ids[i] = candidate_ids[order[i]];
            out_dists[i] = candidate_dists[order[i]];
        }
        return {out_ids, out_dists};
    }

    // remove most stale edges globally; returns count evicted
    int evict(int current_epoch, int n_evict = -1) {
        if (n_evict < 0) {
            int target = (int)(max_total_edges_ * 0.9);
            n_evict = std::max(0, (int)total_edges_ - target);
        }

        if (n_evict <= 0) return 0;

        // collect (staleness, node_id, list_index)
        std::vector<std::tuple<float, uint32_t, size_t>> all_edges;
        for (auto &kv : edges_)
            for (size_t i = 0; i < kv.second.size(); i++)
                all_edges.push_back({staleness(kv.second[i], current_epoch), kv.first, i});

        if (all_edges.empty()) return 0;

        // most stale first
        std::sort(all_edges.begin(), all_edges.end(), [](const auto &a, const auto &b) {
            return std::get<0>(a) > std::get<0>(b);
        });

        size_t n_remove = std::min((size_t)n_evict, all_edges.size());

        std::unordered_map<uint32_t, std::vector<size_t>> by_node;
        for (size_t i = 0; i < n_remove; i++) {
            auto &[stale, node_id, idx] = all_edges[i];
            by_node[node_id].push_back(idx);
        }

        int evicted = 0;
        for (auto &kv : by_node) {
            std::vector<size_t> indices = kv.second;
            std::sort(indices.begin(), indices.end(), std::greater<size_t>());
            std::vector<ConjEdge> &neighbors = edges_[kv.first];
            for (size_t idx : indices) {
                neighbors.erase(neighbors.begin() + idx);
                evicted++;
            }
            if (neighbors.empty()) edges_.erase(kv.first);
        }

        total_edges_ -= evicted;
        return evicted;
    }

    ConjugateGraphStats stats() const {
        if (edges_.empty()) {
            return {0, 0, 0.0, 0, 0.0, 0.0};
        }
        size_t max_edges = 0;
        double sum_edges = 0;
        double sum_traversal = 0;
        double sum_staleness = 0;
        size_t n_edges_total = 0;
        for (auto &kv : edges_) {
            max_edges = std::max(max_edges, kv.second.size());
            sum_edges += kv.second.size();
            for (auto &e : kv.second) {
                sum_traversal += e.traversal_count;
                sum_staleness += staleness(e, 0);
                n_edges_total++;
            }
        }
        ConjugateGraphStats s;
        s.n_nodes_with_edges = edges_.size();
        s.total_edges = total_edges_;
        s.mean_edges_per_node = sum_edges / edges_.size();
        s.max_edges_per_node = max_edges;
        s.mean_traversal_count = n_edges_total ? sum_traversal / n_edges_total : 0.0;
        s.mean_staleness = n_edges_total ? sum_staleness / n_edges_total : 0.0;
        return s;
    }

    // (src, neighbor_id, distance, t_added, epoch_added, traversal_count)
    std::vector<std::tuple<uint32_t, uint32_t, float, float, int, int>> get_all_edges() const {
        std::vector<std::tuple<uint32_t, uint32_t, float, float, int, int>> out;
        for (auto &kv : edges_)
            for (auto &e : kv.second)
                out.push_back({kv.first, e.neighbor_id, e.distance, e.t_added, e.epoch_added, e.traversal_count});
        return out;
    }

    void bulk_load_edges(const std::vector<std::tuple<uint32_t, uint32_t, float, float, int, int>> &all_edges) {
        edges_.clear();
        total_edges_ = 0;
        for (auto &[src, neighbor_id, distance, t_added, epoch_added, traversal_count] : all_edges) {
            edges_[src].push_back(ConjEdge{neighbor_id, distance, t_added, epoch_added, traversal_count});
            total_edges_++;
        }
    }

    size_t M_conj_;
    size_t max_total_edges_;

 private:
    static float squared_l2(const float *a, const float *b, size_t dim) {
        float sum = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    std::unordered_map<uint32_t, std::vector<ConjEdge>> edges_;
    std::vector<ConjEdge> empty_;
    size_t total_edges_;
};

}  // namespace hnswlib
