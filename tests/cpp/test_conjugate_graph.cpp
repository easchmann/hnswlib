// Standalone test for ConjugateGraph — no Python, no pybind11.

#include "../../hnswlib/conjugate_graph.h"

#include <assert.h>

#include <iostream>
#include <vector>

using hnswlib::ConjEdge;
using hnswlib::ConjugateGraph;

namespace {

void test_add_edge_fill_and_evict() {
    // M_conj=2, plenty of global budget
    ConjugateGraph g(2, 1000);

    // both incumbents share epoch_added=1 so the oldest incumbent's epoch_added is 1
    assert(g.add_edge(0, 10, 1.0f, 0.0f, /*epoch_added=*/1, /*current_epoch=*/1));
    assert(g.add_edge(0, 11, 2.0f, 0.0f, /*epoch_added=*/1, /*current_epoch=*/1));
    // node 0 is now full (M_conj=2)

    // same-epoch edge should be rejected (not strictly newer than oldest incumbent's epoch)
    assert(!g.add_edge(0, 12, 3.0f, 0.0f, /*epoch_added=*/1, /*current_epoch=*/1));
    // older-epoch edge should be rejected
    assert(!g.add_edge(0, 13, 3.0f, 0.0f, /*epoch_added=*/0, /*current_epoch=*/1));

    // strictly newer epoch evicts the oldest incumbent (first-inserted, dst=10)
    assert(g.add_edge(0, 14, 4.0f, 0.0f, /*epoch_added=*/2, /*current_epoch=*/2));

    auto all = g.get_all_edges();
    assert(all.size() == 2);
    bool found14 = false, found10 = false;
    for (auto &e : all) {
        if (std::get<1>(e) == 14) found14 = true;
        if (std::get<1>(e) == 10) found10 = true;
    }
    assert(found14);
    assert(!found10);

    std::cout << "test_add_edge_fill_and_evict ok" << std::endl;
}

void test_lookup_traversal_count() {
    ConjugateGraph g(4, 1000);
    g.add_edge(0, 1, 1.0f, 0.0f, 0, 0);
    g.add_edge(0, 2, 1.0f, 0.0f, 0, 0);

    auto &edges1 = g.lookup(0, 0);
    assert(edges1.size() == 2);
    assert(edges1[0].traversal_count == 1);
    assert(edges1[1].traversal_count == 1);

    auto &edges2 = g.lookup(0, 0);
    assert(edges2[0].traversal_count == 2);
    assert(edges2[1].traversal_count == 2);

    // lookup on a node with no edges returns empty, no crash
    auto &edges3 = g.lookup(999, 0);
    assert(edges3.empty());

    std::cout << "test_lookup_traversal_count ok" << std::endl;
}

void test_enhanced_search_hop_asymmetry() {
    ConjugateGraph g(8, 1000);
    // seed(0) -> hop1(1); hop1(1) -> hop2(2)
    g.add_edge(0, 1, 1.0f, 0.0f, 0, 0);
    g.add_edge(1, 2, 1.0f, 0.0f, 0, 0);

    size_t dim = 2;
    std::vector<float> base = {
        0.0f, 0.0f,   // node 0
        1.0f, 0.0f,   // node 1
        2.0f, 0.0f,   // node 2
    };
    std::vector<float> query = {0.0f, 0.0f};

    std::vector<uint32_t> seed_ids = {0};
    std::vector<float> seed_dists = {0.0f};

    auto result = g.enhanced_search(seed_ids, seed_dists, base.data(), dim, query.data(), /*k=*/10, /*current_epoch=*/0, /*two_hop=*/true);

    // node 1's edge (0->1) traversed via hop-1 lookup(): count == 1
    auto edges_from_0 = g.get_all_edges();
    int trav_0_to_1 = -1, trav_1_to_2 = -1;
    for (auto &e : edges_from_0) {
        if (std::get<0>(e) == 0 && std::get<1>(e) == 1) trav_0_to_1 = std::get<5>(e);
        if (std::get<0>(e) == 1 && std::get<1>(e) == 2) trav_1_to_2 = std::get<5>(e);
    }
    assert(trav_0_to_1 == 1);   // hop-1: lookup() called, incremented
    assert(trav_1_to_2 == 0);   // hop-2: raw edges_ access, NOT incremented

    // all three nodes (0 seed, 1 hop1, 2 hop2) should appear in result
    assert(result.first.size() == 3);

    std::cout << "test_enhanced_search_hop_asymmetry ok" << std::endl;
}

void test_enhanced_search_topk_and_padding() {
    ConjugateGraph g(8, 1000);
    // seed node 0 with several hop-1 neighbors at increasing distance
    for (uint32_t i = 1; i <= 5; i++) {
        g.add_edge(0, i, 0.0f, 0.0f, 0, 0);
    }

    size_t dim = 1;
    std::vector<float> base = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};  // node i at value i
    std::vector<float> query = {0.0f};

    std::vector<uint32_t> seed_ids = {0};
    std::vector<float> seed_dists = {0.0f};

    // request k=3 (fewer than the 6 total candidates: seed + 5 hop-1 neighbors)
    auto result = g.enhanced_search(seed_ids, seed_dists, base.data(), dim, query.data(), /*k=*/3, /*current_epoch=*/0, /*two_hop=*/false);
    assert(result.first.size() == 3);
    // sorted ascending by distance: 0 (dist 0), 1 (dist 1), 2 (dist 4)
    assert(result.first[0] == 0);
    assert(result.first[1] == 1);
    assert(result.first[2] == 2);
    for (size_t i = 1; i < result.second.size(); i++) assert(result.second[i - 1] <= result.second[i]);

    // request k=100 (more than the 6 available candidates) -> padding path, all returned sorted
    auto result2 = g.enhanced_search(seed_ids, seed_dists, base.data(), dim, query.data(), /*k=*/100, /*current_epoch=*/0, /*two_hop=*/false);
    assert(result2.first.size() == 6);
    for (size_t i = 1; i < result2.second.size(); i++) assert(result2.second[i - 1] <= result2.second[i]);

    std::cout << "test_enhanced_search_topk_and_padding ok" << std::endl;
}

void test_evict_global_staleness() {
    ConjugateGraph g(8, 1000);
    // src nodes 0, 1, 2 each carry exactly one edge, so lookup() traversal only
    // affects the edge under test (lookup increments ALL edges of a src node).
    g.add_edge(0, 100, 1.0f, 0.0f, /*epoch_added=*/0, 0);  // untouched -> high staleness
    g.add_edge(1, 200, 1.0f, 0.0f, /*epoch_added=*/0, 0);  // heavily traversed -> low staleness
    g.add_edge(2, 300, 1.0f, 0.0f, /*epoch_added=*/9, 9);  // recent -> low staleness

    for (int i = 0; i < 20; i++) g.lookup(1, 9);

    // at current_epoch=10: staleness(0->100) = 10 - 0.5*0  = 10  (highest, never traversed)
    //                       staleness(1->200) = 10 - 0.5*20 = 0
    //                       staleness(2->300) = 1  - 0.5*0  = 1
    // evict exactly 1 edge globally -> must remove 0->100 (highest staleness)
    int evicted = g.evict(/*current_epoch=*/10, /*n_evict=*/1);
    assert(evicted == 1);

    auto all = g.get_all_edges();
    assert(all.size() == 2);
    bool found100 = false, found200 = false, found300 = false;
    for (auto &e : all) {
        if (std::get<1>(e) == 100) found100 = true;
        if (std::get<1>(e) == 200) found200 = true;
        if (std::get<1>(e) == 300) found300 = true;
    }
    assert(!found100);
    assert(found200);
    assert(found300);

    std::cout << "test_evict_global_staleness ok" << std::endl;
}

void test_stats_and_bulk_load() {
    ConjugateGraph g(8, 1000);
    g.add_edge(0, 1, 1.0f, 0.0f, 0, 0);
    g.add_edge(0, 2, 1.0f, 0.0f, 0, 0);
    g.add_edge(3, 4, 1.0f, 0.0f, 0, 0);

    auto s = g.stats();
    assert(s.n_nodes_with_edges == 2);
    assert(s.total_edges == 3);
    assert(s.max_edges_per_node == 2);

    auto all = g.get_all_edges();

    ConjugateGraph g2(8, 1000);
    g2.bulk_load_edges(all);
    auto all2 = g2.get_all_edges();
    assert(all2.size() == all.size());
    auto s2 = g2.stats();
    assert(s2.n_nodes_with_edges == 2);
    assert(s2.total_edges == 3);

    std::cout << "test_stats_and_bulk_load ok" << std::endl;
}

}  // namespace

int main() {
    std::cout << "Testing ConjugateGraph ..." << std::endl;
    test_add_edge_fill_and_evict();
    test_lookup_traversal_count();
    test_enhanced_search_hop_asymmetry();
    test_enhanced_search_topk_and_padding();
    test_evict_global_staleness();
    test_stats_and_bulk_load();
    std::cout << "Test ok" << std::endl;
    return 0;
}
