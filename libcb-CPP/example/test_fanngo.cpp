#include <iostream>
#include <vector>
#include <string>
#include "libcb/ontology.h"
#include "libcb/fanngo.h"
#include "libcb/cafa.h"

using namespace std;
using namespace arma;

// usage:
//
// test_fanngo <ontology file> <training data x> <training data y> <corresp. terms> <test data x>
//
// - <ontology file>    an OBO file of an ontology
// - <training data x>  a csv file of an n-by-m matrix, with n being the number
//                      of sequences and m being the number of features.
// - <training data y>  a csv file of an n-by-p matrix, with k being the number
//                      of terms in the ontology, note that these k terms must
//                      be a subset of the ontology terms specified in the
//                      <ontology file>
// - <corresp. terms>   a list of p terms correspond to the p columns in
//                      <training data y>
// - <test data x>      a csv file of an n'-by-m matrix, with n' being the
//                      number of of test sequences
// - <corresp. seqs>    a list of n' terms correspond to the n' sequences in the
//                      <test data x>
//
// This program trains a FANNGO model and saves it into "./fanngo.model" and
// make predictions on a test file. The prediction is saved into a file
// "fanngo_prediction.cafa" in CAFA format (ready to be evaluated using CAFA2
// matlab evaluation package)

int main(int argc, char* argv[]) {
    // setup
    string modelFile = "fanngo.model";
    string predFile = "fanngo_prediction.cafa";

    // read inputs
    Ontology ontology(argv[1], {"part_of"});
    mat tr_x, tr_y, ts_x, ts_y;
    tr_x.load(argv[2], arma::csv_ascii);
    tr_y.load(argv[3], arma::csv_ascii);
    vector<string> terms = load_items<string>(argv[4]);
    ts_x.load(argv[5], arma::csv_ascii);
    vector<string> test_seqs = load_items<string>(argv[6]);
    
    // setup FANNGO parameters
    FANNGOParam param;
    param.do_subsample_term = true;
    param.num_output = 10;
    param.do_subsample_training = true;
    param.max_sequence = 10000;
    param.do_feature_selection = true;
    param.do_normalization = true;
    param.do_dimension_reduction = true;

    // initialize/train/save a FANNGO model
    FANNGO fanngo(10, ontology, param);
    fanngo.train(tr_x, tr_y);
    ofstream ofs;
    ofs.open(modelFile, ofstream::out);
    fanngo.serialize(ofs);
    ofs.close();

    // predict using the trained model
    ts_y = fanngo.predict(ts_x);

    // output predictions in CAFA format
    cafa_output(predFile, ts_y, test_seqs, terms);
    return 0;
}
