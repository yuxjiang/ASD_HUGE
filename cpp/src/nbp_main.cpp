#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <algorithm>
#include <random>
#include <chrono>

#include <libcb/util.h>
#include <libcb/gene_network.h>
#include <libcb/evaluation.h>

using namespace std;

struct Options {
    // mandatory arguments
    string gfile; // A file of genes.
    string nfile; // A network file in binary format.
    string ofile; // A file of predictions.
    // optional arguments (with their default values)
    nbp::algo_type algo    = nbp::FUNCTIONAL_FLOW; // nbp algorithm
    index_t        nfolds  = 1;                    // The number of folds to cross-validate.
    float          cutoff  = 0.5;                  // The network cutoff.
    float          rewire  = 0.0;                  // The percentage of rewired edges.
    float          shuffle = 0.0;                  // The percentage of shuffled genes.
    index_t        seed    = 0;                    // default seed for random
    // functional-flow specific
    index_t d = 5; // number of iterations
    nbp::FunctionalFlow<Gene>::capacity_type ctype = nbp::FunctionalFlow<Gene>::CAP_RAW;
    // markov random field specific
    float   pi        = 0.0;
    float   alpha     = 0.0;
    float   beta      = 0.0;
    float   gamma     = 0.0;
    index_t burnIn    = 100;
    index_t lagPeriod = 10;
    index_t numSim    = 2000;
} opts;

void usage(void) {
    cerr
        << "NAME\n"
        << "    nbp.run - network-based prediction\n"
        << "\n"
        << "SYNOPSIS\n"
        << "    nbp.run -n <NETWORK FILE> -g <GENE FILE> -o <OUTPUT FILE> [OPTION]...\n"
        << "\n"
        << "DESCRIPTION\n"
        << "    This program runs a network-based prediction algorithm on a network with seeds.\n"
        << "\n"
        << "MANDATORY ARGUMENTS\n"
        << "    -n <NETWORK FILE>\n"
        << "        A network file in BINARY format, i.e., a serialized network.\n"
        << "\n"
        << "    -g <GENE FILE>\n"
        << "        A file of labeld genes with Entrez ID. This file should be a two-column CSV\n"
        << "        <Entrez> <label>\n"
        << "        where <label> must be either 0 (UNKNOWN) or 1 (POSITIVE).\n"
        << "\n"
        << "    -o <OUTPUT FILE>\n"
        << "        An output file name.\n"
        << "\n"
        << "OPTIONAL ARGUMENTS\n"
        << "    -algo <0|1>, default: 0\n"
        << "        The choice of algorithms, must be one of the following:\n"
        << "        0 - functional-flow\n"
        << "        1 - Markov random field\n"
        << "\n"
        << "    -cutoff <Real, non-negative>, default: 0.5\n"
        << "        The network edge weight cutoff.\n"
        << "\n"
        << "    -nfolds <Integer, non-negative>, default: 1\n"
        << "        The number of cross-validation folds.\n"
        << "        If [-nfolds] is set to 0 or 1, it runs in a \"one-time (OT)\" propagation mode.\n"
        << "\n"
        << "    -rewire <Real>, default: 0\n"
        << "        The percentage of edges to rewire. Must be in the interval [0, 1].\n"
        << "        For example, 0.2 means to randomly rewire 20\%|E| pairs of edges.\n"
        << "\n"
        << "    -shuffle <Real>, default: 0\n"
        << "        The percentage of initial genes to shuffle. Must be in the interval [0, 1].\n"
        << "        For example, 0.2 means to randomly shuffle 20\%|positive genes|.\n"
        << "\n"
        << "    -seed <Integer, non-negative>, default: 0\n"
        << "        The seed for pseudo-random number generator.\n"
        << "        0 (default) means to generate a seed from the current time.\n"
        << "        This argument only takes effect when either of the following cases:\n"
        << "        1. [-rewire]  is set to a positve number.\n"
        << "        2. [-shuffle] is set to a positve number.\n"
        << "        3. [-nfolds]  is set to be greater than 2.\n"
        << "\n"
        << "FUNCTIONAL-FLOW SPECIFIC ARGUMENTS\n"
        << "    -ctype <0|1|2>, default: 0\n"
        << "        The capacity type of flow, must be one of the following types:\n"
        << "        0:  raw edge weights\n"
        << "        1:  out-going weights normalized (NO)\n"
        << "        2:  in-coming weights normalized (NI)\n"
        << "    -d <Integer>, default: 5\n"
        << "        The number of functional-flow iterations.\n"
        << "\n"
        << "MARKOV RANDOM FIELD SPECIFIC ARGUMENTS\n"
        << "    -pi <Real, [0, 1]>, default: 0\n"
        << "        If pi is set to 0, an empirical ratio(positives) will be used instead.\n"
        << "    -alpha <Real>, default: 0.0\n"
        << "    -beta  <Real>, default: 0.0\n"
        << "    -gamma <Real>, default: 0.0\n"
        << "    -burn-in <Integer, positive>, default: 100\n"
        << "    -lag-period <Integer, positive>, default: 10\n"
        << "    -nsim <Integer, positive>, default: 2000\n"
        << "\n";
}

void parse_arguments(int argc, const char* argv[]) {
    if (argc <= 1) {
        usage();
        exit(1);
    }
    int i = 1;
    while (i < argc) {
        if (string(argv[i]).compare("-g") == 0) {
            opts.gfile = argv[++i];
            if (!has_file(opts.gfile)) {
                error_and_exit(__PRETTY_FUNCTION__, "Gene file does not exist.");
            }
        } else if (string(argv[i]).compare("-n") == 0) {
            opts.nfile = argv[++i];
            if (!has_file(opts.nfile)) {
                error_and_exit(__PRETTY_FUNCTION__, "Network file does not exist.");
            }
        } else if (string(argv[i]).compare("-o") == 0) {
            opts.ofile = argv[++i];
        } else if (string(argv[i]).compare("-cutoff") == 0) {
            opts.cutoff = stod(argv[++i]);
            if (opts.cutoff < 0) {
                usage();
                error_and_exit(__PRETTY_FUNCTION__, "Cutoff should be >= 0.");
            }
        } else if (string(argv[i]).compare("-nfolds") == 0) {
            index_t nfolds = stoi(argv[++i]);
            if (nfolds < 0) {
                usage();
                error_and_exit(__PRETTY_FUNCTION__, "Number of cross-validation folds must be >= 0.");
            }
            if (nfolds > 10) {
                warn(__PRETTY_FUNCTION__, "Number of cross-validation folds >= 10.");
            }
            opts.nfolds = nfolds;
        } else if (string(argv[i]).compare("-rewire") == 0) {
            opts.rewire = stod(argv[++i]);
            if (opts.rewire < 0 || opts.rewire > 1) {
                usage();
                error_and_exit(__PRETTY_FUNCTION__, "Rewire percentage must be in the interval [0, 1].");
            }
        } else if (string(argv[i]).compare("-shuffle") == 0) {
            opts.shuffle = stod(argv[++i]);
            if (opts.shuffle < 0 || opts.shuffle > 1) {
                usage();
                error_and_exit(__PRETTY_FUNCTION__, "Shuffle percentage must be in the interval [0, 1].");
            }
        } else if (string(argv[i]).compare("-algo") == 0) {
            int choice = stoi(argv[++i]);
            if (choice == 0) {
                opts.algo = nbp::FUNCTIONAL_FLOW;
            } else if (choice == 1) {
                opts.algo = nbp::MARKOV_RANDOM_FIELD;
            } else {
                usage();
                error_and_exit(__PRETTY_FUNCTION__, "Unknown capacity type.");
            }
        } else if (string(argv[i]).compare("-seed") == 0) {
            index_t seed = stoi(argv[++i]);
            if (seed < 0) {
                usage();
                error_and_exit(__PRETTY_FUNCTION__, "Seed should be non-negative.");
            }
            opts.seed = seed;
        } else if (string(argv[i]).compare("-d") == 0) {
            opts.d = stoi(argv[++i]);
        } else if (string(argv[i]).compare("-ctype") == 0) {
            int choice = stoi(argv[++i]);
            if (choice == 0) {
                opts.ctype = nbp::FunctionalFlow<Gene>::CAP_RAW;
            } else if (choice == 1) {
                opts.ctype = nbp::FunctionalFlow<Gene>::CAP_NO;
            } else if (choice == 2) {
                opts.ctype = nbp::FunctionalFlow<Gene>::CAP_NI;
            } else {
                usage();
                error_and_exit(__PRETTY_FUNCTION__, "Unknown capacity type.");
            }
        } else if (string(argv[i]).compare("-pi") == 0) {
            opts.alpha = stod(argv[++i]);
        } else if (string(argv[i]).compare("-alpha") == 0) {
            opts.alpha = stod(argv[++i]);
        } else if (string(argv[i]).compare("-beta") == 0) {
            opts.beta = stod(argv[++i]);
        } else if (string(argv[i]).compare("-gamma") == 0) {
            opts.beta = stod(argv[++i]);
        } else if (string(argv[i]).compare("-burn-in") == 0) {
            opts.burnIn = static_cast<unsigned>(stoi(argv[++i]));
        } else if (string(argv[i]).compare("-lag-period") == 0) {
            opts.lagPeriod = static_cast<unsigned>(stoi(argv[++i]));
        } else if (string(argv[i]).compare("-nsim") == 0) {
            opts.numSim = static_cast<unsigned>(stoi(argv[++i]));
        } else {
            usage();
            error_and_exit(__PRETTY_FUNCTION__, "Unknown option.");
        }
        i += 1;
    }
    // check mandatory arguments
    if (opts.gfile.empty()) {
        usage();
        error_and_exit(__PRETTY_FUNCTION__, "No gene file specified.");
    }
    if (opts.nfile.empty()) {
        usage();
        error_and_exit(__PRETTY_FUNCTION__, "No network file specified.");
    }
    if (opts.ofile.empty()) {
        usage();
        error_and_exit(__PRETTY_FUNCTION__, "No output file specified.");
    }
}

void parse_gene_file(const string& file, vector<Gene>& positives, vector<Gene>& unknowns) {
    ifstream ifs(file, ifstream::in);
    positives.resize(0);
    unknowns.resize(0);
    string buf;
    string entrez;
    int label;
    while (ifs.good()) {
        getline(ifs, buf);
        if (buf.empty()) continue;
        istringstream iss(buf);
        iss >> entrez >> label;
        if (label == 1) {
            positives.emplace_back(entrez);
        } else {
            unknowns.emplace_back(entrez);
        }
    }
    ifs.close();
}

void normalize_predictions(unordered_map<Gene, nbp::score_type>& predictions) {
    nbp::score_type max_score(0);
    for (const auto& prediction : predictions) {
        if (max_score < prediction.second) {
            max_score = prediction.second;
        }
    }
    if (max_score > 0) {
        for (auto& prediction : predictions) {
            prediction.second /= max_score;
        }
    }
}

int main(int argc, const char* argv[]) {
    parse_arguments(argc, argv);

    log_message(__PRETTY_FUNCTION__, "Start preprocessing.");

    // load gene labels
    vector<Gene> positives, unknowns, genes;
    parse_gene_file(opts.gfile, positives, unknowns);
    if (opts.shuffle > 0) {
        // shuffle positives <-> unknowns
        int numberOfSwaps = std::max(1, static_cast<int>(opts.shuffle * positives.size())); // at least swap one pair
        unsigned seed = opts.seed;
        if (seed == 0) {
            seed = std::chrono::system_clock::now().time_since_epoch().count();
        }
        std::shuffle(positives.begin(), positives.end(), std::default_random_engine(seed));
        std::shuffle(unknowns.begin(), unknowns.end(), std::default_random_engine(seed));
        for (int i = 0; i < numberOfSwaps; ++i) {
            std::swap(positives[i], unknowns[i]);
        }
    }
    genes = positives;
    genes.insert(genes.end(), unknowns.begin(), unknowns.end());
    // load gene networks
    SGraph<Gene> gn;
    ifstream ifs(opts.nfile, ifstream::in);
    gn.deserialize(ifs);
    ifs.close();
    // filter edges and rewire if necessary
    gn.filter_edges(opts.cutoff);
    if (opts.rewire > 0) {
        unsigned long ne = gn.number_of_edges();
        gn = degree_preserving_edge_rewire(gn, static_cast<unsigned long>(ne * opts.rewire));
    }

    log_message(__PRETTY_FUNCTION__, "Gene network configured.");

    // select algorithm
    // nbp::FunctionalFlow ff(opts.ctype, opts.d);
    shared_ptr<nbp::Algorithm<Gene>> algoptr;
    switch (opts.algo) {
        case nbp::FUNCTIONAL_FLOW:
            algoptr = make_shared<nbp::FunctionalFlow<Gene>>(
                    opts.ctype,
                    opts.d);
            break;
        case nbp::MARKOV_RANDOM_FIELD:
            algoptr = make_shared<nbp::MarkovRandomField<Gene>>(
                    opts.pi,
                    opts.alpha,
                    opts.beta,
                    opts.gamma,
                    opts.burnIn,
                    opts.lagPeriod,
                    opts.numSim);
            break;
        case nbp::DUMMY:
        default:
            error_and_exit(__PRETTY_FUNCTION__, "Unknown algorithm.");
            break;
    }
    unordered_map<Gene, nbp::score_type> predictions;
    if (opts.nfolds <= 1) { // direct prediction mode
        predictions = (*algoptr)(gn, positives);
        normalize_predictions(predictions);
        // Set positives to be all 1.
        for (const auto& positive : positives) {
            predictions[positive] = 1.0;
        }
    } else { // cross-validation mode
        CVFolds pos_folds(positives.size(), opts.nfolds, opts.seed);
        CVFolds unk_folds(unknowns.size(), opts.nfolds, opts.seed);
        // initialize `predictions` to be all 0
        for (const auto& gene : genes) {
            predictions[gene] = 0.0;
        }
        unordered_map<Gene, nbp::score_type> infold_pred;
        for (index_t v = 0; v < opts.nfolds; ++v) {
            // trainging = positive training fold
            // test = positive test fold + unknown test fold
            auto pos_tr = pos_folds.training_fold(v);
            auto pos_ts = pos_folds.test_fold(v);
            auto unk_ts = unk_folds.test_fold(v);
            vector<Gene> training(pos_tr.size());
            for (size_t i = 0; i < pos_tr.size(); ++i) {
                training[i] = positives[pos_tr[i]];
            }
            vector<Gene> test(pos_ts.size() + unk_ts.size());
            for (size_t i = 0; i < pos_ts.size(); ++i) {
                test[i] = positives[pos_ts[i]];
            }
            for (size_t i = 0; i < unk_ts.size(); ++i) {
                test[i + pos_ts.size()] = unknowns[unk_ts[i]];
            }
            infold_pred = (*algoptr)(gn, training);
            // collect predictions on test set
            for (const auto& gene : test) {
                if (CONTAINS(infold_pred, gene)) {
                    predictions[gene] = infold_pred.at(gene);
                }
            }
        }
        normalize_predictions(predictions);
    }
    // output
    ofstream ofs(opts.ofile, ofstream::out);
    for (const auto& gene : genes) {
        if (CONTAINS(predictions, gene) && predictions.at(gene) > 0) {
            ofs << gene.id << "\t" << predictions.at(gene) << endl;
        }
    }
    ofs.close();
}

// -------------
// Yuxiang Jiang (yuxjiang@indiana.edu)
// Department of Computer Science
// Indiana University, Bloomington
// Last modified: Fri 04 Sep 2020 06:28:56 PM E
