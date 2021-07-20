//! libcb/gene_network.cpp

#include <iostream>
#include <iomanip>
#include <cmath>

#include "../include/libcb/util.h"
#include "../include/libcb/gene_network.h"

using namespace std;

GeneNetwork load_gene_network_by_nodes(const string& nFile) {
    vector<Gene> genes;
    vector<string> ids = load_items<string>(nFile);
    for (const auto& id : ids) {
        genes.emplace_back(id);
    }
    return GeneNetwork(genes);
}

GeneNetwork load_gene_network_by_edges(const string& eFile, char delim) {
    vector<Gene> genes;
    vector<GeneNetwork::weighted_edge_type> edges;
    unordered_map<string, index_t> id2index;
    string buf, src, dst, wbuf;
    float weight;
    ifstream ifs(eFile, ifstream::in);
    index_t indexedCount = 0;
    while (ifs.good()) {
        getline(ifs, buf);
        if (buf.empty()) continue; // skip empty lines
        istringstream iss(buf);
        try {
            getline(iss, src, delim);
            getline(iss, dst, delim);
            getline(iss, wbuf, delim);
            weight = stod(wbuf);
            // iss >> src >> dst >> weight;
            // cerr << "[debug] reading " << src << "\t" << dst << "\t" << weight << "\n";
            if (!CONTAINS(id2index, src)) {
                id2index[src] = indexedCount++;
                genes.emplace_back(src);
                cerr << "[debug] added gene " << src << " -> " << id2index.at(src) << "\n";
            }
            if (!CONTAINS(id2index, dst)) {
                id2index[dst] = indexedCount++;
                genes.emplace_back(dst);
                cerr << "[debug] added gene " << dst << " -> " << id2index.at(dst) << "\n";
            }
            index_t u = id2index.at(src);
            index_t v = id2index.at(dst);
            GeneNetwork::weight_type w = fabs(static_cast<GeneNetwork::weight_type>(weight));
            edges.emplace_back(u, v, w);
        } catch (...) {
            cerr << "[warning] skip non-numeric weight\n";
        }

    }
    ifs.close();
    return GeneNetwork(genes, edges);
}

GeneNetwork load_gene_network_by_nodes_and_edges(
        const string& nFile,
        const string& eFile,
        char delim) {
    vector<Gene> genes;
    vector<GeneNetwork::weighted_edge_type> edges;
    unordered_map<string, index_t> id2index;
    vector<string> ids = load_items<string>(nFile);
    for (size_t i = 0; i < ids.size(); ++i) {
        id2index[ids[i]] = i;
        genes.emplace_back(ids[i]);
    }
    string buf, src, dst, wbuf;
    float weight;
    ifstream ifs(eFile, ifstream::in);
    while (ifs.good()) {
        getline(ifs, buf);
        if (buf.empty())
            continue;
        istringstream iss(buf);
        try {
            getline(iss, src, delim);
            getline(iss, dst, delim);
            getline(iss, wbuf, delim);
            weight = stod(wbuf);
            // iss >> src >> dst >> weight;
            if (!CONTAINS(id2index, src) || !CONTAINS(id2index, dst))
                continue;
            index_t u = id2index.at(src);
            index_t v = id2index.at(dst);
            GeneNetwork::weight_type w = fabs(static_cast<GeneNetwork::weight_type>(weight));
            edges.emplace_back(u, v, w);
        } catch (...) {
            cerr << "[warning] skip non-numeric weight\n";
        }
    }
    ifs.close();
    return GeneNetwork(genes, edges);
}

void save_gene_network(const string& eFile,
        const GeneNetwork& gn) {
    ofstream ofs(eFile, ofstream::out);
    vector<Gene> genes = gn.get_vertices();
    vector<GeneNetwork::weighted_edge_type> edges = gn.get_weighted_edges();
    index_t u, v;
    GeneNetwork::weight_type w;
    for (const auto& edge : edges) {
        tie(u, v, w) = edge;
        ofs << genes[u].id << "\t" << genes[v].id << "\t" << w << "\n";
    }
    ofs.close();
}

void print_small_gene_network(const GeneNetwork& gn) {
    vector<Gene> genes = gn.get_vertices();
    // find printing column width
    size_t maxlen_id(0); // maximum length of Entrez ID
    for (const auto& gene : genes) {
        if (maxlen_id < gene.id.length()) {
            maxlen_id = gene.id.length();
        }
    }
    size_t col_width = maxlen_id > 6 ? maxlen_id : 6; // width must be at least 6
    // print header line of the table
    cout << setfill(' ') << setw(col_width + 1) << "";
    for (index_t i = 0; i < gn.number_of_vertices(); ++i) {
        cout << setfill(' ') << setw(col_width + 1) << genes[i].id;
    }
    cout << "\n";
    for (index_t r = 0; r < gn.number_of_vertices(); ++r) {
        cout << setfill(' ') << setw(col_width + 1) << genes[r].id;
        for (index_t c = 0; c < gn.number_of_vertices(); ++c) {
            if (gn.has_edge(r, c)) {
                cout << setprecision(3)
                    << fixed
                    << setw(col_width + 1)
                    << gn.get_edge_weight(r, c);
            } else {
                cout << setprecision(1)
                    << fixed
                    << setw(col_width + 1)
                    << "-";
            }
        }
        cout << "\n";
    }
}

// -------------
// Yuxiang Jiang (yuxjiang@indiana.edu)
// Department of Computer Science
// Indiana University, Bloomington
// Last modified: Tue 08 Jan 2019 10:42:14 AM P
