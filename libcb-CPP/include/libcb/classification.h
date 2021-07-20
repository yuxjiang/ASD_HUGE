#ifndef _LIBCB_MODEL_H_
#define _LIBCB_MODEL_H_

#include "util.h"
#include "data_matrix.h"

// classification model
struct Model {
  Model() {} // creates a default model
  virtual void train(const arma::mat& x, const arma::mat& y) = 0;

  virtual arma::mat predict(const arma::mat& x) const = 0;

  virtual void print() const = 0;
  virtual ~Model() {}

  virtual void serialize(std::ostream& stream) const = 0;
  virtual void deserialize(std::istream& stream) = 0;

  const Model& operator=(const Model& other) {
    if (this != &other) {
      num_examples = other.num_examples;
      num_features = other.num_features;
      num_targets  = other.num_targets;
    }
    return *this;
  }

  /**
   * Number of examples that are attached to this model
   * (Note: this includes training and possibly validation examples)
   */
  index_t num_examples;

  /**
   * Number of features of this model
   */
  index_t num_features;

  /**
   * Number of target variables of this model
   */
  index_t num_targets;
};

// a bagging ensemble
template<typename T>
struct Ensemble : public Model {
  using model_type = T;

  Ensemble(index_t n = 0) { resize(n); }

  inline index_t size() const {
    return models.size();
  }

  void resize(index_t n) {
    models.resize(n);
    sampled_examples.resize(n);
    sampled_targets.resize(n);
  }

  /**
   * @brief trains an ensemble model.
   *
   * @remark thie function implements the base class Model::train()
   */
  virtual void train(const arma::mat& x, const arma::mat& y) {
    train_oob(x, y, false);
  }

  /**
   * @brief trains an ensemble model and output the out-of-bag prediction.
   */
  virtual arma::mat train_oob(const arma::mat& x, const arma::mat& y, bool oob) {
    using namespace std;
    using namespace arma;

    num_examples = x.n_rows;
    num_features = x.n_cols;
    num_targets  = y.n_cols;
    if (x.n_rows != y.n_rows) {
      error_and_exit(__PRETTY_FUNCTION__, "Inconsistent number of examples.");
    }

    size_t k = models.size();
#pragma omp parallel for
    for (size_t i = 0; i < k; ++i) {
      // This function is usually to be overridden by the derived classes so as to setup
      // sampled_examples and sampled_targets for the i-th model accordingly
      train_setup_i(i, x, y);
      models[i].train(x.rows(sampled_examples[i]), y(sampled_examples[i], sampled_targets[i]));
#pragma omp critical
      {
        models[i].print();
        std::cout << "[INFO] finished training model " << i + 1 << "/" << k << "\n";
      }
    }
    mat oob_pred; // place-holder
    if (oob) {
      oob_pred = zeros<mat>(arma::size(y));
      mat counts(arma::size(y), fill::zeros);
      uvec universe = regspace<uvec>(0, num_examples - 1);
      size_t k = models.size();
      for (size_t i = 0; i < k; ++i) {
        uvec bagged = sort(sampled_examples[i]);
        uvec not_bagged(num_examples, fill::zeros);
        auto it = set_difference(universe.begin(),
            universe.end(),
            bagged.begin(),
            bagged.end(),
            not_bagged.begin());
        not_bagged.resize(it - not_bagged.begin());
        // ...
        oob_pred(not_bagged, sampled_targets[i]) += models[i].predict(x.rows(not_bagged));
        counts(not_bagged, sampled_targets[i]) += 1.0;
      }
      uvec predicted = find(counts > datum::eps);
      oob_pred(predicted) = oob_pred(predicted) / counts(predicted);
      // oob_pred(find(counts <= datum::eps)).fill(datum::nan);
    }
    return oob_pred;
  }

  virtual arma::mat predict(const arma::mat& x) const {
    using namespace std;
    using namespace arma;
    if (x.n_cols != num_features) {
      error_and_exit(__PRETTY_FUNCTION__, "Inconsistent number of features.");
    }
    mat y = zeros<mat>(x.n_rows, num_targets);
    vec counts = zeros<vec>(num_targets);
    size_t k = models.size();
#pragma omp parallel for
    for (size_t i = 0; i < k; ++i) {
      mat sy = models[i].predict(x);
#pragma omp critical
      {
        // accumulate results;
        y.cols(sampled_targets[i]) += sy;
        counts(sampled_targets[i]) += 1;
        cout << "[INFO] predicted on model " << i + 1 << "/" << k << "\n";
      }
    }
    // averaging
    for (index_t j = 0; j < num_targets; ++j) {
      y.col(j) /= counts(j);
    }
    return y;
  }

  /**
   * @brief sets up training data and model for i-th model.
   *
   * @remark this function provides the basic setup for training the i-th model of the ensemble,
   * i.e. to make a bootstrapped training sample. Derived classes usually override this function.
   */
  virtual void train_setup_i(index_t i, const arma::mat& x, const arma::mat& y) {
    index_t n = x.n_rows;
    index_t p = y.n_cols;
    sampled_examples[i] = randsample(n, n, true);   // make a bootstrap sample
    sampled_targets[i]  = arma::regspace<arma::uvec>(0, p - 1); // keep all targets
  }

  void serialize(std::ostream& stream) const {
    uint32_t data;

    data = models.size();
    SAVE_VAR(stream, data);

    data = num_examples;
    SAVE_VAR(stream, data);

    data = num_features;
    SAVE_VAR(stream, data);

    data = num_targets;
    SAVE_VAR(stream, data);

    for (index_t i = 0; i < models.size(); ++i) {
      models[i].serialize(stream);
      sampled_examples[i].save(stream);
      sampled_targets[i].save(stream);
    }
  }

  void deserialize(std::istream& stream) {
    uint32_t data;

    LOAD_N(stream, data, sizeof(uint32_t));
    resize(data);

    LOAD_N(stream, data, sizeof(uint32_t));
    num_examples = data;

    LOAD_N(stream, data, sizeof(uint32_t));
    num_features = data;

    LOAD_N(stream, data, sizeof(uint32_t));
    num_targets = data;

    for (index_t i = 0; i < models.size(); ++i) {
      models[i].deserialize(stream);
      sampled_examples[i].load(stream);
      sampled_targets[i].load(stream);
    }
  }

  inline void print() const {
    std::cout << "Ensemble model." << std::endl;
  }

  const Ensemble<T>& operator=(const Ensemble<T>& other) {
    Model::operator=(other);
    if (this != &other) {
      resize(other.size());
      for (index_t i = 0; i < models.size(); ++i) {
        models[i]           = other.models[i];
        sampled_examples[i] = other.sampled_examples[i];
        sampled_targets[i]  = other.sampled_targets[i];
      }
    }
    return *this;
  }

  std::vector<model_type> models;
  std::vector<arma::uvec> sampled_examples;
  std::vector<arma::uvec> sampled_targets;
};

#endif // _LIBCB_MODEL_H_

// -------------
// Yuxiang Jiang (yuxjiang@indiana.edu)
// Department of Computer Science
// Indiana University, Bloomington
// Last modified: Sun Aug 11 16:18:29 2019
