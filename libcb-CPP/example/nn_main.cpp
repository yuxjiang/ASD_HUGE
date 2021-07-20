#include <iostream>
#include "libcb/util.h"
#include "libcb/fanngo.h"
#include "libcb/evaluation.h"
#include "libcb/classification.h"
#include <armadillo>

using namespace std;
using namespace arma;

/**
 * This program does a simple sanity check for the following modules
 * (warning: not a full test, only for core functionalities)
 * - cross validation
 * - ensemble
 * - neural network
 * - load/save for ensemble and neural network
 *
 * The expected accuracy on iris should be very close to 1.
 */

int main(int argc, char* argv[]) {
    Ensemble<NeuralNetwork> nn_ensemble(1);
    // configure each neural network model
    for (index_t i = 0; i < nn_ensemble.size(); ++i) {
        nn_ensemble.models[i].train_algo = NeuralNetwork::TRAIN_GD;
        nn_ensemble.models[i].minibatch_size = 0;
        nn_ensemble.models[i].early_stopping = false;
    }
    mat x, y, y2;
    x.load("/Users/yuxjiang/Data/parsed/housing_X.csv", raw_ascii);
    y.load("/Users/yuxjiang/Data/parsed/housing_y.csv", raw_ascii);

    index_t n = x.n_rows; // number of examples
    // index_t m = x.n_cols; // number of features
    index_t p = static_cast<index_t>(arma::max(arma::max(y)));
    index_t v = 5; // number of cv folds

    // convert y into one-hot encoding
    y2 = zeros<mat>(n, p);
    for (index_t i = 0; i < n; ++i) {
        y2(i, static_cast<index_t>(y(i)) - 1) = 1.0;
    }

    // shuffle data points
    uvec indices = shuffle(regspace<uvec>(0, n - 1));
    CVFolds cv(n, v);
    ofstream ofs;
    ifstream ifs;
    for (index_t i = 0; i < v; ++i) {
        auto tr = uvec(cv.training_fold(i));
        auto ts = uvec(cv.test_fold(i));
        mat tr_x = x.rows(indices(tr));
        mat tr_y = y2.rows(indices(tr));
        mat ts_x = x.rows(indices(ts));
        mat ts_y = y.rows(indices(ts));

        nn_ensemble.train(tr_x, tr_y);

        ofs.open("model" + to_string(i) + ".model", ofstream::out);
        nn_ensemble.serialize(ofs);
        ofs.close();

        ifs.open("model" + to_string(i) + ".model", ifstream::in);
        Ensemble<NeuralNetwork> replicate;
        replicate.deserialize(ifs);
        ifs.close();

        uvec predicted = index_max(replicate.predict(ts_x), 1) + 1;

        // calculate acc
        index_t ts_n = ts.n_elem;
        index_t count = 0;
        for (index_t i = 0; i < ts_n; ++i) {
            if (static_cast<index_t>(ts_y(i)) == predicted(i)) {
                count ++;
            }
        }

        // printing
        cout << "fold: " << i + 1 << " accuracy: " << (double)(count) / ts_n << endl;
    }
    return 0;
}
