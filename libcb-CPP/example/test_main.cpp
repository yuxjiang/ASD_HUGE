#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include "../include/libcb/util.h"
#include "../include/libcb/gene_network.h"
#include "../include/libcb/fanngo.h"

using namespace std;

void print_small_fmat(const vector<vector<float>>&);
void print_small_graph_as_fmat(const SGraph<Gene>&);
void test_degree_of(const SGraph<Gene>&, const Gene&);
void test_neighbor_of(const SGraph<Gene>&, const Gene&);
void test_singleton(SGraph<Gene>&, const Gene&);
void test_connected_component(const SGraph<Gene>&);
void test_get_functions(const SGraph<Gene>&);
void test_set_functions(SGraph<Gene>&, const Gene&, const Gene&, float);
void test_hubs(const SGraph<Gene>&);
void test_basics(const SGraph<Gene>&);
void test_subgraph(const SGraph<Gene>&);
void test_rewire(const SGraph<Gene>&, unsigned long);
void test_merge(const SGraph<Gene>&, const SGraph<Gene>&, const string&);
void test_save(const SGraph<Gene>&, const string&);
void test_fanngo_param(const string& filename);

int main(int argc, char* argv[]) {
    // SGraph<Gene> gn = load_gene_network_by_edges(argv[1]);
    // Gene gene(argv[2]);
    // print_small_gene_network(gn);
    // test_set_functions(gn, gene);
    // test_rewire(gn, stoi(argv[2]));
    // Gene gsrc(argv[2]), gdst(argv[3]);
    // float w = stod(argv[4]);
    // test_set_functions(gn, gsrc, gdst, w);
    // SGraph<Gene> gn2 = load_gene_network(argv[2]);
    // print_small_gene_network(gn2);
    // test_merge(gn, gn2, argv[3]);
    // test_save(gn, argv[2]);
    test_fanngo_param(argv[1]);
    return 0;
}

void print_small_fmat(const vector<vector<float>>& x) {
    size_t n = x.size(); if (n == 0) return;
    size_t m = x[0].size(); if (m == 0) return;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            if (x[i][j] > 0) {
                cout << setprecision(3) << fixed << setw(7) << x[i][j];
            } else {
                cout << setprecision(3) << fixed << setw(7) << "-";
            }
        }
        cout << endl;
    }
}

void print_small_graph_as_fmat(const SGraph<Gene>& gn) {
    vector<vector<float>> x(gn.number_of_vertices());
    for (auto& row : x) {
        row.resize(gn.number_of_vertices(), 0);
    }
    for (index_t i = 0; i < gn.number_of_vertices(); ++i) {
        for (index_t j = i + 1; j < gn.number_of_vertices(); ++j) {
            x[i][j] = x[j][i] = gn.get_edge_weight(i, j);
        }
    }
    print_small_fmat(x);
}

void test_degree_of(const SGraph<Gene>& gn, const Gene& gene) {
    vector<Gene> genes = gn.get_vertices();
    unordered_map<Gene, index_t> g2i;
    for (size_t i = 0; i < genes.size(); ++i) {
        g2i[genes[i]] = i;
    }
    if (!CONTAINS(g2i, gene)) {
        error_and_exit("test_degree_of()", "invalid gene");
    }
    cout << "degree of " << gene.id << " is: " << gn.degree_of(g2i.at(gene))
        << endl;
}

void test_neighbor_of(const SGraph<Gene>& gn, const Gene& gene) {
    vector<Gene> genes = gn.get_vertices();
    unordered_map<Gene, index_t> g2i;
    for (size_t i = 0; i < genes.size(); ++i) {
        g2i[genes[i]] = i;
    }
    if (!CONTAINS(g2i, gene)) {
        error_and_exit("test_neighbor_of()", "invalid gene");
    }
    vector<index_t> neighbors = gn.neighbor_of(g2i.at(gene));
    cout << "neighbor of " << gene.id << " is:";
    for (auto const& neighbor : neighbors) {
        cout << " " << genes[neighbor].id;
    }
    cout << endl;
}

void test_singleton(SGraph<Gene>& gn, const Gene& gene) {
    cout << "number of singletons: " << gn.number_of_singletons() << endl;
    vector<Gene> genes = gn.get_vertices();
    unordered_map<Gene, index_t> g2i;
    for (size_t i = 0; i < genes.size(); ++i) {
        g2i[genes[i]] = i;
    }
    if (!CONTAINS(g2i, gene)) {
        error_and_exit("test_singleton()", "invalid gene");
    }
    gn.disconnect(g2i.at(gene));
    cout << "after disconnecting " << gene.id << endl;
    print_small_gene_network(gn);
    gn.remove_singletons();
    cout << "after removing singletons " << endl;
    print_small_gene_network(gn);
}

void test_connected_component(const SGraph<Gene>& gn) {
    cout << "number of connected component: " << gn.number_of_connected_component() << endl;
}

void test_get_functions(const SGraph<Gene>& gn) {
    vector<Gene> genes = gn.get_vertices();
    cout << "number of vertices: " << gn.number_of_vertices() << endl;
    cout << "number of edges: " << gn.number_of_edges() << endl;
    vector<SGraph<Gene>::weighted_edge_type> weighted_edges = gn.get_weighted_edges();
    vector<SGraph<Gene>::edge_type> edges = gn.get_edges();
    cout << "all weighted edges: " << endl;
    index_t src, dst;
    SGraph<Gene>::weight_type weight;
    for (auto const& we : weighted_edges) {
        std::tie(src, dst, weight) = we;
        cout << "(" << genes[src].id << ", " << genes[dst].id << ", "
            << weight << ")" << endl;
    }
    cout << "all edges (with get_edge_weight()): " << endl;
    for (auto const& e : edges) {
        std::tie(src, dst) = e;
        cout << "(" << genes[src].id << ", " << genes[dst].id << ") -> "
            << gn.get_edge_weight(src, dst) << endl;
    }
}

void test_set_functions(SGraph<Gene>& gn, const Gene& gsrc, const Gene& gdst, const float weight) {
    vector<Gene> genes = gn.get_vertices();
    unordered_map<Gene, index_t> g2i;
    for (size_t i = 0; i < genes.size(); ++i) {
        g2i[genes[i]] = i;
    }
    if (CONTAINS(g2i, gsrc) && CONTAINS(g2i, gdst)) {
        if (weight > 0) {
            gn.set_edge_weight(g2i.at(gsrc), g2i.at(gdst), weight);
        } else {
            gn.remove_edge(g2i.at(gsrc), g2i.at(gdst));
        }
    } else {
        error_and_exit("test_set_functions()", "invalid gene");
    }
    cout << "after setting edges" << endl;
    print_small_gene_network(gn);
}

void test_hubs(const SGraph<Gene>& gn) {
    vector<Gene> genes = gn.get_vertices();
    vector<index_t> hubs = gn.get_hubs(3);
    cout << "hubs:" << endl;
    for (auto const& hub : hubs) {
        cout << genes[hub].id << "(" << gn.degree_of(hub) << ")" << endl;
    }
}

void test_basics(const SGraph<Gene>& gn) {
    SGraph<Gene> new_gn = gn;
    cout << "This is a new graph created by operator=" << endl;
    print_small_gene_network(new_gn);
    cout << "after resized" << endl;
    new_gn.resize(9);
    print_small_gene_network(new_gn);
}

void test_subgraph(const SGraph<Gene>& gn) {
    auto n = gn.number_of_vertices();
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<index_t> distribution(0, n - 1);
    unordered_set<index_t> index_set;
    while (index_set.size() < (n / 2)) {
        index_set.insert(distribution(generator));
    }
    vector<index_t> indices;
    for (auto const& index : index_set) {
        indices.push_back(index);
    }
    SGraph<Gene> subgn = gn.subgraph(indices);
    cout << "after rewired" << endl;
    print_small_gene_network(subgn);
}

void test_rewire(const SGraph<Gene>& gn, unsigned long n) {
    SGraph<Gene> new_gn = degree_preserving_edge_rewire(gn, n);
    print_small_gene_network(new_gn);
}

void test_merge(const SGraph<Gene>& gna, const SGraph<Gene>& gnb, const string& sch) {
    SGraph<Gene> g;
    if (sch.compare("max") == 0) {
        g = gna.merge(gnb, SGraph<Gene>::merge_scheme::maximum);
    } else if (sch.compare("min") == 0) {
        g = gna.merge(gnb, SGraph<Gene>::merge_scheme::minimum);
    } else if (sch.compare("add") == 0) {
        g = gna.merge(gnb, SGraph<Gene>::merge_scheme::addition);
    } else {
        error_and_exit("test_merge()", "unknown scheme");
    }
    cout << "merged network" << endl;
    print_small_gene_network(g);
}

void test_save(const SGraph<Gene>& gn, const string& filename) {
    save_gene_network(filename, gn);
}

void test_fanngo_param(const string& filename) {
    arma::arma_rng::set_seed_random();
    // test save/load fanngo parameter
    FANNGOParam param;
    param.max_sequence = 12345;
    param.max_output = 135;
    param.do_feature_selection = true;
    param.selected_features = arma::regspace<arma::uvec>(0, 2, 8);
    param.do_normalization = false;
    param.mus = arma::randu<arma::mat>(1, 5);
    param.sigmas = arma::randu<arma::mat>(1, 5);
    param.do_dimension_reduction = true;
    param.ret_ratio = 0.56;
    param.coeff = arma::randu<arma::mat>(4, 5);

    cout << "saving ..." << endl;
    param.print();
    ofstream ofs;
    ofs.open(filename, ofstream::out);
    param.serialize(ofs);
    ofs.close();

    ifstream ifs;
    ifs.open(filename, ifstream::in);
    FANNGOParam param2;
    param2.deserialize(ifs);
    ifs.close();
    cout << "loading ..." << endl;
    param2.print();
}
// end of file

// -------------
// Yuxiang Jiang (yuxjiang@indiana.edu)
// Department of Computer Science
// Indiana University, Bloomington
// Last modified: Thu 11 Apr 2019 05:09:25 PM P
