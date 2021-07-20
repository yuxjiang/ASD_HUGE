#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <libcb/evaluation.h>

using namespace std;

void usage() {
    cerr
        << "NAME\n"
        << "    get-auc.run - get area under the ROC curve\n"
        << "\n"
        << "SYNOPSIS\n"
        << "    get-auc.run <LABEL FILE> <PREDICTION FILE>\n"
        << "\n"
        << "DESCRIPTION\n"
        << "    This program computes AUC of a prediction.\n"
        << "\n"
        << "    Both files should be two-column CSV files, delimited by <SPACE>\n"
        << "    <Entrez ID> <label/score>\n"
        << "\n";
}

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        usage();
        exit(1);
    }
    ifstream ifs;
    // read labels
    ifs.open(argv[1], ifstream::in);
    string buf;
    vector<string> tags;
    vector<bool>   labels;
    while (ifs.good()) {
        getline(ifs, buf);
        if (buf.empty()) continue;
        string tag;
        bool label;
        istringstream iss(buf);
        iss >> tag >> label;
        tags.push_back(tag);
        labels.push_back(label);
    }
    ifs.close();
    // read predictions
    ifs.open(argv[2], ifstream::in);
    unordered_map<string, float> predictions;
    while (ifs.good()) {
        getline(ifs, buf);
        if (buf.empty()) continue;
        string tag;
        float score;
        istringstream iss(buf);
        iss >> tag >> score;
        predictions[tag] = score;
    }
    ifs.close();
    // align
    vector<float> scores(tags.size(), 0.0);
    for (size_t i = 0; i < tags.size(); ++i) {
        if (CONTAINS(predictions, tags[i])) {
            scores[i] = predictions.at(tags[i]);
        }
    }
    cout << get_auc<float>(scores, labels) << endl;
    return 0;
}
