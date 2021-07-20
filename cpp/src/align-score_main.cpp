#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>

#include "libcb/util.h"
using namespace std;

struct {
    string gfile;
    string sfile;
    string ofile;
    string pfile = "";
    string nfile = "";
} opts;

void usage() {
    cerr
        << "NAME\n"
        << "    align-score.run - align prediction score.\n"
        << "\n"
        << "SYNOPSIS\n"
        << "    align-score.run -g <GENE FILE> -s <SCORE FILE> -o <OUTPUT FILE> [OPTION]..\n"
        << "\n"
        << "DESCRIPTION\n"
        << "    This program aligns a predictions to a specific gene list.\n"
        << "    Note that unscored genes are set to be zeros.\n"
        << "\n"
        << "MANDATORY ARGUMENTS\n"
        << "    -g <GENE FILE>\n"
        << "        A file of gene list\n"
        << "\n"
        << "    -s <SCORE FILE>\n"
        << "        A prediction file as a two-column CSV, delimited by TAB.\n"
        << "        <Entrez> <score>\n"
        << "\n"
        << "    -o <OUTPUT FILE>\n"
        << "        An output file name.\n"
        << "\n"
        << "OPTIONAL ARGUMENTS\n"
        << "    -pos <POS FILE>, default: EMPTY\n"
        << "        A file of positive gene list that you want to them to be clamped as 1.\n"
        << "\n"
        << "    -neg <NEG FILE>, default: EMPTY\n"
        << "        A file of negative gene list that you want to them to be clamped as 0.\n"
        << "\n";
}

void parseArguments(int argc, const char* argv[]) {
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
        } else if (string(argv[i]).compare("-s") == 0) {
            opts.sfile = argv[++i];
            if (!has_file(opts.sfile)) {
                error_and_exit(__PRETTY_FUNCTION__, "Score file does not exist.");
            }
        } else if (string(argv[i]).compare("-o") == 0) {
            opts.ofile = argv[++i];
        } else if (string(argv[i]).compare("-pos") == 0) {
            opts.pfile = argv[++i];
            if (!has_file(opts.pfile)) {
                error_and_exit(__PRETTY_FUNCTION__, "Positive gene file does not exist.");
            }
        } else if (string(argv[i]).compare("-pos") == 0) {
            opts.nfile = argv[++i];
            if (!has_file(opts.nfile)) {
                error_and_exit(__PRETTY_FUNCTION__, "Negative gene file does not exist.");
            }
        } else {
            usage();
            error_and_exit(__PRETTY_FUNCTION__, "Unknown option.");
        }
        i += 1;
    }
    if (opts.gfile.empty()) {
        usage();
        error_and_exit(__PRETTY_FUNCTION__, "No gene file specified.");
    }
    if (opts.sfile.empty()) {
        usage();
        error_and_exit(__PRETTY_FUNCTION__, "No network file specified.");
    }
    if (opts.ofile.empty()) {
        usage();
        error_and_exit(__PRETTY_FUNCTION__, "No output file specified.");
    }
}

int main(int argc, const char* argv[]) {
    parseArguments(argc, argv);

    vector<string> entrezs = load_items<string>(opts.gfile);
    unordered_map<string, float> prediction;
    for (const auto& entrez : entrezs) {
        prediction[entrez] = 0.0;
    }

    ifstream ifs(opts.sfile, iostream::in);
    while (ifs.good()) {
        string buf;
        getline(ifs, buf);
        if (buf.empty()) continue;
        string e;
        float s;
        istringstream iss(buf);
        iss >> e >> s;
        if (CONTAINS(prediction, e)) {
            prediction[e] = s;
        }
    }
    ifs.close();

    // clamp scores if necessary
    if (!opts.pfile.empty()) {
        vector<string> posEntrezs = load_items<string>(opts.pfile);
        for (const auto& posEntrez : posEntrezs) {
            if (CONTAINS(prediction, posEntrez)) {
                prediction[posEntrez] = 1.0;
            }
        }
    }
    if (!opts.nfile.empty()) {
        vector<string> negEntrezs = load_items<string>(opts.nfile);
        for (const auto& negEntrez : negEntrezs) {
            if (CONTAINS(prediction, negEntrez)) {
                prediction[negEntrez] = 0.0;
            }
        }
    }

    ofstream ofs(opts.ofile, iostream::out);
    for (const auto& entrez : entrezs) {
        ofs << setprecision(8) << std::fixed << entrez << "\t" << prediction[entrez] << "\n";
    }
    ofs.close();
    return 0;
}
