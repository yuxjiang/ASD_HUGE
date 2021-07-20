#include "../include/libcb/gene_network.h"

using namespace std;

int main(int argc, const char* argv[]) {
    // load an ASCII edge network
    SGraph<Gene> gn = load_gene_network_by_edges(argv[1]);
    // load a vector of positive genes
    vector<string> entrezs = load_items<string>(argv[2]);
    vector<Gene> genes(entrezs.size());
    for (size_t i = 0; i < entrezs.size(); ++i) {
        genes[i].id = entrezs[i];
    }
    // run functional flow
    nbp::FunctionalFlow<Gene> functionalflow;
    functionalflow.set_d(3);
    unordered_map<Gene, nbp::score_type> predictions = functionalflow(gn, genes);
    // print
    for (auto const& prediction : predictions) {
        Gene gene;
        nbp::score_type score;
        tie(gene, score) = prediction;
        cout << gene.id << ": " << score << endl;
    }
    return 0;
}
