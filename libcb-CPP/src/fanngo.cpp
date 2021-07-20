#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <random>
#include <vector>
#include <chrono>
#include <omp.h>
#include <cstdint>
#include <cmath>

#include "../include/libcb/util.h"
#include "../include/libcb/distribution.h"
#include "../include/libcb/fanngo.h"

#define NUM_THREADS 16

using namespace std;
using namespace arma;

// FANNGOParam
void FANNGOParam::print() const {
  cout << "FANNGO parameter:\n";
  cout << "-----------------\n";
  cout << "maximum sequence: " << max_sequence << endl;
  cout << "maximum output:   " << max_output << endl;
  cout << "Feature selection? ";
  if (do_feature_selection) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  cout << endl;
  cout << "selected columns:";
  for (index_t i = 0; i < selected_features.n_elem; ++i) {
    cout << " " << selected_features(i);
  }
  cout << endl;

  cout << "Normalization? ";
  if (do_normalization) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  cout << endl;
  mus.print("mu = ");
  sigmas.print("sigma = ");

  cout << "Dimension reduction? ";
  if (do_dimension_reduction) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  cout << endl;
  cout << "Retained variance: " << ret_ratio << endl;
  coeff.print("coefficient = ");
}

void FANNGOParam::serialize(ostream& stream) const {
  uint32_t data;

  data = max_sequence;
  SAVE_VAR(stream, data);

  data = max_output;
  SAVE_VAR(stream, data);

  data = do_feature_selection;
  SAVE_VAR(stream, data);
  selected_features.save(stream);

  data = do_normalization;
  SAVE_VAR(stream, data);
  mus.save(stream);
  sigmas.save(stream);

  data = do_dimension_reduction;
  SAVE_VAR(stream, data);
  SAVE_VAR(stream, ret_ratio);
  coeff.save(stream);
}

void FANNGOParam::deserialize(istream& stream) {
  uint32_t data;
  LOAD_N(stream, data, sizeof(uint32_t));
  max_sequence = data;

  LOAD_N(stream, data, sizeof(uint32_t));
  max_output = data;

  LOAD_N(stream, data, sizeof(uint32_t));
  do_feature_selection = data != 0;
  selected_features.load(stream);

  LOAD_N(stream, data, sizeof(uint32_t));
  do_normalization = data != 0;
  mus.load(stream);
  sigmas.load(stream);

  LOAD_N(stream, data, sizeof(uint32_t));
  do_dimension_reduction = data != 0;
  LOAD_N(stream, ret_ratio, sizeof(double));
  coeff.load(stream);
}

// FANNGOSingle
void FANNGOSingle::_prepare_ont_training(const arma::mat& y) {
  // build a backward index from string to index in the sub-ontology[i]
  unordered_map<string, index_t> reverse_index;
  auto terms = ontology.terms();
  for (index_t i = 0; i < terms.size(); ++i) {
    reverse_index[terms[i]] = i;
  }

  _training_masks.clear();
  for (index_t i = 0; i < y.n_rows; ++i) {
    set<index_t> indices;
    set<string> sampled_terms;
    for (index_t j = 0; j < y.n_cols; ++j) {
      if (y(i, j) > 1e-6) {
        sampled_terms.insert(terms[j]);
      }
    }
    // add annotated leaves
    auto leaves = ontology.leaves_of(sampled_terms);
    for (const auto& leaf: leaves) {
      indices.insert(reverse_index[leaf]);
    }

    // add children of these leaves
    auto children = ontology.children(leaves);
    for (const auto& child: children) {
      indices.insert(reverse_index[child]);
    }

    // For examples that has less than 10 selected target to update, make it 10 randomly
    index_t min_num_target = 10;
    if (indices.size() < min_num_target) {
      arma::arma_rng::set_seed_random();
      uvec shuffled_index = randsample(y.n_cols, y.n_cols, false); // shuffle indices
      index_t curr = 0;
      while (indices.size() < min_num_target && curr < shuffled_index.size()) {
        indices.insert(shuffled_index(curr++));
      }
    }

    // append indices of selected target of example i to _training_masks
    uvec mask(indices.size()); 
    index_t cnt = 0;
    for (const auto& index : indices) {
      mask(cnt++) = index;
    }
    _training_masks.emplace_back(mask);
  }
}

void FANNGOSingle::_feature_selection(arma::mat& x, const arma::mat& y) {
  index_t budget = 5;
  double p_sig   = 1e-10; // significant level for calling a "good" feature
  if (x.n_cols <= budget) {
    // select all features/columns if less than the "budget"
    param.selected_features = find(ones<uvec>(x.n_cols));
    return;
  }
  uvec selected = zeros<uvec>(x.n_cols);
  for (index_t k = 0; k < y.n_cols; ++k) {
    // for each target column in y, select at least "budget" features
    uvec this_sel = zeros<uvec>(x.n_cols);
    for (index_t j = 0; j < x.n_cols; ++j) {
      vec this_col = x.col(j);
      // make two samples and then t-test
      vec x0 = this_col(find(y.col(k) == 0));
      vec x1 = this_col(find(y.col(k) == 1));
      if (x0.is_empty() || x1.is_empty()) {
        this_sel(j) = 0;
      } else if (arma::mean(x0) == arma::mean(x1)) {
        this_sel(j) = 0;
      } else if (arma::stddev(x0) == 0 && arma::stddev(x1) == 0) {
        this_sel(j) = 1;
      } else if (htest::ttest2(x0, x1) < p_sig) {
        this_sel(j) = 1;
      } else {
        this_sel(j) = 0;
      }
    }
    index_t n_sel = arma::sum(this_sel);
    if (n_sel < budget) {
      // randomly select at least "budget" features/columns of x
      uvec left = find(this_sel == 0);
      uvec extra = randsample(left, budget - n_sel, false);
      for (index_t j = 0; j < extra.n_elem; ++j) {
        this_sel(extra(j)) = 1;
      }
    }
    selected += this_sel; // collect votes
  }
  param.selected_features = find(selected > 0);
  // apply the selected features
  x = x.cols(param.selected_features);

  // #ifdef ENABLE_LOGGING
  //     cerr << "[debug] feature selection for network [" << i << "]: "
  //         << n_cols << " -> " << x.n_cols << "\n";
  // #endif

}

void FANNGOSingle::_normalization(arma::mat& x) {

  param.mus    = arma::mean(x);
  param.sigmas = arma::stddev(x); // normalization type using N-1 by armadillo default
  for (index_t j = 0; j < x.n_cols; ++j) {
    if (param.sigmas(0, j) < datum::eps) {
      // standard deviation too small (possibly zero), then set it to be one.
      param.sigmas(0, j) = 1.0;
    }
    x.col(j) = (x.col(j) - param.mus(0, j)) / param.sigmas(0, j);
  }
}

void FANNGOSingle::_dimension_reduction(arma::mat& x) {
  mat cov_hat = x.t() * x;
  vec eig_val;
  mat eig_vec;
  eig_sym(eig_val, eig_vec, cov_hat);
  // flip eigenvalues and the corresponding eigenvector, since by default,
  // Armadillo returns eigenvalues in ascending order.
  eig_val = flipud(eig_val);
  eig_vec = fliplr(eig_vec);

  // cerr << "[debug] has_nan? " << x.has_nan() << "\n";
  // cerr << "[debug] size of x is: (" << x.n_rows << ", " << x.n_cols << ")\n";
  // cerr << "[debug] size of cov(x) is: (" << cov_hat.n_rows << ", " << cov_hat.n_cols << ")\n";

  // eig_val.print("eigval =");
  // eig_vec.print("eigvec =");
  // vec temp = cumsum(eig_val)/sum(eig_val);
  // temp.t().print("cumsum ratio = ");
  uvec first_encounter = find((cumsum(eig_val)/sum(eig_val) >= param.ret_ratio), 1, "first");
  // cerr << "[debug] Okay?\n";
  // cerr << "[debug] size of \"first_encounter\" is: " << first_encounter.n_elem << "\n";
  param.coeff = eig_vec.cols(0, first_encounter(0));
  // cerr << "[debug] YES!\n";
  x = x * param.coeff;

  // #ifdef ENABLE_LOGGING
  //   cerr
  //     << "[debug] dimension reduction for network [" << i << "]: "
  //     << n_cols << " -> " << x.n_cols << "\n";
  // #endif

}

arma::mat FANNGOSingle::predict(const arma::mat& x) const {
  mat x_ = x;
  if (param.do_feature_selection) {
    x_ = x_.cols(param.selected_features);
  }
  // normalization and pca
  if (param.do_normalization) {
    for (index_t j = 0; j < x_.n_cols; ++j) {
      x_.col(j) = (x_.col(j) - param.mus(0, j)) / param.sigmas(0, j);
    }
  }
  if (param.do_dimension_reduction) {
    x_ = x_ * param.coeff; // dimension: (n x m) * (m x l)
  }
  /**
   * Depending on the loss function of the network, we can determine which model it is:
   * loss     model
   * ----     -----
   * MSE      fanngo
   * MSECON   confanngo
   * SD       confanngo_sd
   */
  mat pred;
  if (is_consistent()) {
    pred = _ell->consistent_output(NeuralNetwork::predict(x_).t()).t();
  } else {
    pred = NeuralNetwork::predict(x_);
  }
  return pred;
}

void FANNGOSingle::train(const arma::mat& x, const arma::mat& y) {
  mat x_ = x, y_ = y; // make local copies of x and y
  if (param.max_sequence > 0 && param.max_sequence < x_.n_rows) {
    uvec index = randsample(x_.n_rows, param.max_sequence, false);
    x_ = x.rows(index);
    y_ = y.rows(index);
  }
  if (param.do_training_subsampling) {
    _prepare_ont_training(y_);
  }
  if (param.do_feature_selection) {
    _feature_selection(x_, y_);
  }
  if (param.do_normalization) {
    _normalization(x_);
  }
  if (param.do_dimension_reduction) {
    _dimension_reduction(x_);
  }
  NeuralNetwork::train(x_, y_);
}

// FANNGO
void FANNGO::train_setup_i(index_t i, const arma::mat& x, const arma::mat& y) {
  models[i].param = param; // pass down the parameters to each FANNGOSingle
  index_t n = x.n_rows;
  index_t p = y.n_cols;
  index_t k = param.max_output;
  sampled_examples[i] = randsample(n, n, true); // make a bootstrap sample

  // make a subontology and then attach to this FANNGOSingle model; sample without replacement a
  // subset of target terms and keep them in the order of the topological order of the subontology.
  if (k > 0 && k < p) {
    // build a back index for terms
    vector<string> terms = ontology.terms();
    unordered_map<string, index_t> terms_back_index;
    for (index_t j = 0; j < terms.size(); ++j) {
      terms_back_index[terms[j]] = j;
    }
    // subsample the attached ontology to the i-th model, then update it
    sampled_targets[i] = randsample(p, k, false);

    set<string> sampled_terms;
    for (index_t j = 0; j < sampled_targets[i].size(); ++j) {
      sampled_terms.insert(terms[sampled_targets[i](j)]);
    }
    models[i].ontology = ontology.subontology(sampled_terms);

    // re-order sampled_targets[i] to align with the topological order returned by Ontology::terms()
    auto ordered_sampled_terms = models[i].ontology.terms();
    for (index_t j = 0; j < ordered_sampled_terms.size(); ++j) {
      sampled_targets[i](j) = terms_back_index[ordered_sampled_terms[j]];
    }
  } else {
    // include all terms, no need to update ontology
    sampled_targets[i] = regspace<uvec>(0, p - 1);
    models[i].ontology = ontology;
  }

  // Update loss function parameters according pre-set loss function.
  // Note that when setting the loss function during model initialization, the parameters (e.g. the
  // ancestor matrix A()) is usually not set (i.e. set as a placeholder) so each of the fanngo model
  // in the ensemble is not fully configured (target node is not sampled yet) until this function
  // being called during train().
  if (models[i].is_consistent()) {
    // if enforcing consistency
    vector<arma::mat> params;
    params.push_back(models[i].ontology.A());
    models[i].set_loss_function(models[i].get_loss_function(), params);
  }

}

void FANNGO::combine(const FANNGO& other) {
  if (ontology == other.ontology) {
    index_t n = models.size();
    index_t m = other.size();
    resize(n + m); // expand
    for (index_t i = 0; i < m; ++i) {
      models[n + i] = other.models[i];
      sampled_examples[n + i] = other.sampled_examples[i];
      sampled_targets[n + i]  = other.sampled_targets[i];
    }
  } else {
    warn(__PRETTY_FUNCTION__, "Ontology mismatch, skip combination.");
    // error_and_exit(__PRETTY_FUNCTION__, "Ontology mismatch.");
  }
}

arma::mat FANNGO::predict(const arma::mat& x) const {
  if (x.n_cols != num_features) {
    error_and_exit(__PRETTY_FUNCTION__, "Inconsistent number of features.");
  }
  mat y = zeros<mat>(x.n_rows, num_targets);
  size_t k = models.size();
  vector<string> terms = ontology.terms();
  // build up the index of targets (terms) in the entire (original) target set
  unordered_map<string, index_t> term_index;
  for (index_t i = 0; i < terms.size(); ++i) {
    term_index[terms[i]] = i;
  }
#pragma omp parallel for
  for (size_t i = 0; i < k; ++i) {
    // predict on sampled terms
    mat sampled_y = models[i].predict(x);

    // fill predicted sampled columns into the large matrix wait to be propagated
    mat y_ = zeros(x.n_rows, num_targets); // y_ is a temp matrix holding propagated predictions
    y_.cols(sampled_targets[i]) = sampled_y;

    // propagate in the reverse topological order
    for (auto it = terms.rbegin(); it != terms.rend(); ++it) {
      for (const auto& parent : ontology.parents(*it)) {
        y_.col(term_index[parent]) = arma::max(y_.col(term_index[parent]), y_.col(term_index[*it]));
      }
    }
    y_ /= (models[i].ontology.A() * arma::ones(sampled_targets[i].n_elem, 1)).max();
#pragma omp critical
    {
      // accumulate results;
      // TODO: we might need a better way to normalize the final score from each model
      // y += (y_ / y_.max());
      y += y_;
      cout << "[INFO] predicted on model " << i + 1 << "/" << k << "\n";
    }
  }
  // return the average
  return y / k;
}
