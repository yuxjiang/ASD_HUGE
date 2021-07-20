/**
 * Notes on gradients
 * ====
 *
 * Dimensions
 * ----
 *  u = 0, 1, 2 ... h-1 the index of layers. In particular,
 *
 *  u = 0:        the input layer (features)
 *  u = 1 .. h-2: the hidden layers
 *  u = h-1:      the output layer (predictions)
 *
 *  k[u]    the dimension of the u-th layer (number of neurons)
 *
 *  w[u]    the weights matrix connecting the (u-1)-th and the u-th layer.
 *          dim(w[u]) = k[u] x k[u-1] (with w[0] not being used)
 *
 *  b[u]    the weights vector connecting the bias node from the (u-1)-th layer to each node in the
 *          u-th layer.
 *          dim(b[u]) = k[u] x 1 (with b[0] not being used)
 *
 *  g[u]    the weighted linear summation from the (u-1)-th layer
 *          dim(g[u]) = k[u] x 1 (with g[0] not being used)
 *
 *  f[u]    the output from the u-th layer. dim(f[u]) = k[u] x 1
 *
 * Computation on the u-th layer (starting from u = 1)
 * ----
 *  f[u] = transfer( g[u] + b[u] )
 *       = transfer( w[u] * f[u-1] + b[u] )
 *
 * Other variables
 * ----
 *  h       the number of layers, h >= 3, note that the index of the last layer is: h-1.
 *
 *  l       the loss function, l: R^k[h-1] --> R, i.e., l is a single-valued multivariate function.
 *
 *  df[u]   { partial(l) / partial(f[u]) }^T
 *          dim(df[u]) = 1 x k[u] (with df[0] not being used)
 *
 *  dg[u]   { partial(f[u]) / partial(g[u]) }
 *          technically, dim(dg[u]) = k[u] x k[u]. However, as the transfer (activation) function is
 *          an element-wise function, since it's just a diagonal matrix, we only store its diagonal
 *          as a column vector, dim(dg[u]) = k[u] x 1
 *          (with dg[0] not being used)
 *
 *  dw[u]   partial(l) / partial(w[u]), dim(dw[u]) = k[u] x k[u-1]
 *          { partial(l) / partial(w[u]_ij) }
 *          note that we set dim(dw[u]) = dim(w[u]) to simplify updating, it should be transposed
 *          technically.
 *          (with dw[0] not being used)
 *
 *  db[u]   partial(l) / partial(b[u]), dim(db[u]) = k[u] x 1
 *          { partial(l) / partial(d[u]_i) }
 *          Same as dw[u] dim(db[u]) = dim(b[u]) for easy updating technically, it should be 1 x
 *          k[u], as a row vector.
 *          (with db[0] not being used)
 *
 * Gradients
 * ----
 * Element-wise version
 *    To compute dw[u]_ij ('%' for element-wise product)
 *
 *    dw[u]_ij = (df[u] % dg[u]^T)_i * f[u-1]_j, and
 *    db[u]_i  = df[u]_i
 *
 *    thus, to make them more compact,
 *
 * Compact form (matrix form)
 *    dw[u] = (dg[u] % df[u]^T) * f[u-1]^T
 *    db[u] = df[u]^T
 *
 * Note that df[u] can be computed recursively:
 * df[u] = (df[u+1] % dg[u+1]^T) * w[u+1]
 *
 * Double-check on dimensions:
 * (1 x k[u]) =  (1 x k[u+1]) % (k[u+1] x 1)^T * (k[u+1] x k[u])
 *
 * Back-propagation algorithm (implementation)
 * ----
 * Initialization:
 * 1. feedforward, and update the following
 *    g[u], for u = 1 .. (h-1)
 *    f[u], for u = 0 .. (h-1)
 *
 * 2. df[h-1] (depends on the type of loss function)
 *
 * 3. dg[h-1] (depends on the transfer function on layer h-1)
 *
 * 4. db[h-1] (simply a copy of df[h-1]^T)
 *
 * Execution:
 * for u <- (h-2) .. 1
 *   df[u] <- (df[u+1] % dg[u+1]^T) * w[u+1]
 *   dg[u] <- {update dg[u] for a given transfer function on layer u}
 *   dw[u] <- (dg[u] % df[u]^T) * f[u-1]^T
 *   db[u] <- df[u]^T
 *
 *   w[u] <- w[u] - eta * dw[u]
 *   b[u] <- b[u] - eta * db[u]
 *
 * Final notes on notation: indices used in coding
 * ----
 *  u    layer index
 *  i    row index
 *  j    col index
 *  n    total number of data points
 *  ii   index of data points
 */
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <map>
#include <utility>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>

#include "../include/libcb/util.h"
#include "../include/libcb/data_matrix.h"
#include "../include/libcb/neural_network.h"

using namespace std;
using namespace arma;

void NeuralNetwork::resize(const vector<index_t>& hidden_layers) {
  if (hidden_layers.empty()) {
    error_and_exit(__PRETTY_FUNCTION__, "Requires at least 1 hidden layer.");
  }
  _k = vector<index_t>(hidden_layers.size() + 2);
  for (size_t i = 1; i < _k.size() - 1; ++i) {
    _k[i] = hidden_layers[i - 1];
  }
  _h = _k.size(); // set the number of layers

  // initialize internal matrices
  _f.resize(_h); _df.resize(_h);
  _g.resize(_h); _dg.resize(_h);
  _w.resize(_h); _dw.resize(_h); _deltaw.resize(_h); _optimalw.resize(_h);
  _b.resize(_h); _db.resize(_h); _deltab.resize(_h); _optimalb.resize(_h);
  // transfer functions
  _tfcns.resize(_h); _tfcns_id.resize(_h);

  _x.clear(); _vx.clear();
  _y.clear(); _vy.clear();
  _n = _vn = 0;
  _init();
}

void NeuralNetwork::_init() {
  if (_h > 0) {
    // reset status
    _net_status = STATUS_OKAY;
    // the default training algorithm
    train_algo = TRAIN_RPROP;
    // the default loss function: mse
    _ell = make_shared<MSE>(MSE());
    // the default transfer functions: sigmoid
    for (index_t u = 0; u < _h; ++u) {
      _tfcns[u]    = make_shared<Sigmoid>(Sigmoid());
      _tfcns_id[u] = TF_SIGMOID;
    }
    // the default validation proportion
    _vp = 0.25f;

    // additional resets
    _training_masks.clear();

    // the default training parameters
    early_stopping = true;
    eta            = 1e-2; // This could be better!
    min_diff       = 1e-8;
    min_grad       = 1e-7;
    max_fails      = 25;
    goal           = 0;
    epochs         = 5000;
    minibatch_size = 0; // batch mode
  }
}

void NeuralNetwork::train(const arma::mat& x, const arma::mat& y) {
  if (x.n_rows != y.n_rows) {
    error_and_exit(__PRETTY_FUNCTION__, "Inconsistent number of examples.");
  }
  num_examples = x.n_rows;
  num_features = x.n_cols;
  num_targets  = y.n_cols;
  if (!early_stopping) {
    _vp = 0.0;
  }
  // _attach_data() will partition the training and validation set if needed and reset (_x, _y, _n)
  // and (_vx, _vy, _vn) accordingly.
  _attach_data(x, y);

  // determine minibatch size
  minibatch_size = minibatch_size >= _n ? 0 : minibatch_size;

  // initialize parameters before training
  _train_init();
  switch (train_algo) {
    case TRAIN_GD:
      if (minibatch_size == 0) {
        _train_gd();
      } else if (minibatch_size == 1) {
        _train_sgd();
      } else {
        _train_mgd();
      }
      break;
    case TRAIN_RMSPROP:
      if (minibatch_size == 0) {
        warn(__PRETTY_FUNCTION__, "minibatch size unset, switch to rprop algorithm instead.");
        train_algo = TRAIN_RPROP;
        _train_rprop();
      } else {
        _train_rmsprop();
      }
      break;
    case TRAIN_RPROP:
    default:
      _train_rprop(); break;
  }
  // use the "optimal" model on the validation partition.
  if (early_stopping && _vn > 0) {
    _w = _optimalw;
    _b = _optimalb;
  }
}

arma::mat NeuralNetwork::predict(const arma::mat& x) const {
  // check dimensions
  if (x.n_cols != _k.front()) {
    cerr << "input features: " << x.n_cols << ", network dim: " << _k.front() << endl;
    error_and_exit(__PRETTY_FUNCTION__, "Incorrect number of features.");
  }
  // Note that we assume the input matrix x is [n x m], i.e.,
  // each data point corresp. to a row.
  mat gu, f = x.t();
  for (index_t u = 1; u < _h; ++u) {
    gu = _w[u] * f;
    gu.each_col() += _b[u]; // for each data point, in-place addition of bias
    f = _tfcns[u]->transfer(gu);
  }
  return f.t();
}

void NeuralNetwork::_attach_data(const arma::mat& x, const arma::mat& y) {
  _k.front() = x.n_cols;
  _k.back()  = y.n_cols;
  // pour data into local training x, and y, note that the dimension of ds.x
  // and ds.y should be [n x m] and [n x p] so that we need to transpose here.
  _x = x.t();
  _y = y.t();
  _n = x.n_rows;
  _make_validation(); // make the validation partition if needed.
}

void NeuralNetwork::_make_validation() {
  if (_vp > 0) {
    // split for the validation set
    dset_t ds = dset_t(dmat_t(_x.t()), dmat_t(_y.t()));
    vector<dset_t> dss;
    vector<uvec> indices; // the index of example for the [0]:training, [1]:validation partition
    std::tie(dss, indices) = ds.partition(1 - _vp, _vp);
    _x  = dss[0].x.data.t();
    _y  = dss[0].y.data.t();
    _n  = dss[0].num_examples();
    _vx = dss[1].x.data.t();
    _vy = dss[1].y.data.t();
    _vn = dss[1].num_examples();

    // keep the training partition only if necessary, i.e., indices[0]
    if (!_training_masks.empty()) {
      vector<uvec> masks;
      for (index_t i = 0; i < indices[0].size(); ++i) {
        masks.emplace_back(_training_masks[indices[0](i)]);
      }
      _training_masks = masks;
    }
  }
}

void NeuralNetwork::print() const {
  string algo_tag, status_tag;

  switch (train_algo) {
    case TRAIN_GD:
      if (minibatch_size == 0) {
        algo_tag = "batch gradient descent";
      } else if (minibatch_size == 1) {
        algo_tag = "stochastic gradient descent";
      } else {
        algo_tag = "minibatch gradient descent";
      }
      break;
    case TRAIN_RPROP:
      algo_tag = "resilient propagation";
      break;
    case TRAIN_RMSPROP:
      algo_tag = "root mean square propagation";
      break;
    default:
      algo_tag = "unknown";
  }

  switch (_net_status) {
    case STATUS_OKAY:              status_tag = "running"; break;
    case STATUS_ERROR:             status_tag = "error"; break;
    case STATUS_CONVERGED:         status_tag = "converged"; break;
    case STATUS_REACHED_GOAL:      status_tag = "reached minimum loss"; break;
    case STATUS_REACHED_MAX_EPOCH: status_tag = "maximum epoch number reached"; break;
    case STATUS_REACHED_MAX_FAIL:  status_tag = "maximum fail number reached"; break;
    default:                       status_tag = "unknown"; break;
  }

  cout << endl << left;

  string token  = "[INFO]";
  string title  = "Neural Network Information";
  size_t ntoken = token.length() + 1;
  size_t ntitle = title.length();
  size_t width  = 16; // the width of tag name of each line

  string section, section_tag;

  // section: data set
  section = "data sets";
  section_tag = " " + section + " ";
  cout << token << " " << title << endl;
  cout
    << setw(ntoken) << ""
    << string(ntitle, '-').replace(2, section_tag.length(), section_tag) << endl;
  cout
    << setw(ntoken) << ""
    << setw(width) << "training:"   << _n  << endl;
  cout
    << setw(ntoken) << ""
    << setw(width) << "validation:" << _vn << endl;

  // section: structure
  section = "structure";
  section_tag = " " + section + " ";
  cout
    << setw(ntoken) << ""
    << string(ntitle, '-').replace(2, section_tag.length(), section_tag) << endl;
  cout
    << setw(ntoken) << ""
    << setw(width) << "input:" << _k.front() << " neurons." << endl;
  for (index_t u = 1; u < _h - 1; ++u) {
    cout
      << setw(ntoken) << ""
      << setw(width) << string("layer[]:").insert(6, to_string(u))
      << _k[u] << " neurons." << endl;
  }
  cout
    << setw(ntoken) << ""
    << setw(width) << "output:" << _k.back() << " neurons." << endl;

  // section: config
  section = "config";
  section_tag = " " + section + " ";
  cout
    << setw(ntoken) << ""
    << string(ntitle, '-').replace(2, section_tag.length(), section_tag) << endl;
  cout
    << setw(ntoken) << ""
    << setw(width) << "algorithm:" << algo_tag << endl;
  if (minibatch_size > 0) {
    cout
      << setw(ntoken) << ""
      << setw(width) << "minibatch:" << minibatch_size << endl;
  }
  cout
    << setw(ntoken) << ""
    << setw(width) << "loss:" << _ell->name << endl;
  for (index_t u = 1; u < _h; ++u) {
    cout
      << setw(ntoken) << ""
      << setw(width) << string("transfer[]:").insert(9, to_string(u))
      << _tfcns[u]->name << endl;
  }

  // section: config
  section = "probes";
  section_tag = " " + section + " ";
  cout
    << setw(ntoken) << ""
    << string(ntitle, '-').replace(2, section_tag.length(), section_tag) << endl;
  cout
    << setw(ntoken) << ""
    << setw(width) << "status:"
    << status_tag << endl;

  cout
    << setw(ntoken) << ""
    << setw(width) << "loss:"
    << _loss << endl;
  cout
    << setw(ntoken) << ""
    << setw(width) << "delta(loss):"
    << std::fabs(_loss - _loss_prev) << endl;
  cout
    << setw(ntoken) << ""
    << setw(width) << "num of epoch:"
    << _epoch << endl;
  if (early_stopping && _vx.n_cols > 0) {
    cout
      << setw(ntoken) << ""
      << setw(width) << "vloss:"
      << _vloss << endl;
    cout
      << setw(ntoken) << ""
      << setw(width) << "vloss*:"
      << _vloss_min << endl;
    cout
      << setw(ntoken) << ""
      << setw(width) << "num of fails:"
      << _fails << endl;
  }
  cout << setw(ntoken + ntitle) << string(ntitle, '-') << endl;
}

void NeuralNetwork::print_debug() const {
  cout << endl;

  string token  = "[DEBUG] ";
  string title  = "Neural Network Information";
  size_t ntoken = token.length();
  size_t ntitle = title.length();

  cout << token << title << endl;
  cout << string(ntoken, ' ') << string(ntitle, ' ') << endl;
  cout << string(ntoken, ' ') << "number of data points: " << _n << endl;
  for (index_t u = 0; u < _h; ++u) {
    cout << string(ntoken, ' ') << "layer[" << u << "]: " << _k[u] << endl;
  }
  cout << string(ntoken, ' ') << "--------------------------" << endl;
  cout << string(ntoken, ' ') << "dimensions:" << endl;
  for (index_t u = 1; u < _h; ++u) {
    cout << string(ntoken, ' ') << "layer[" << u - 1 << "->" << u << "]:";
    cout << string(ntoken, ' ') << "w[" << u << "]: " << _w[u].n_rows << "x" << _w[u].n_cols << ", "
      << string(ntoken, ' ') << "b[" << u << "]: " << _b[u].n_rows << "x" << _b[u].n_cols << endl;
  }

  cout << string(ntoken, ' ') << "--------------------------" << endl;
  cout << string(ntoken, ' ') << "weights:" << endl;
  for (index_t u = 1; u < _h; ++u) {
    string wu = string("w[") + to_string(u) + string("] =");
    string bu = string("b[") + to_string(u) + string("] =");
    _w[u].print(wu.c_str());
    _b[u].print(bu.c_str());
    cout << endl;
  }
  cout << string(ntoken, ' ') << "--------------------------" << endl;
}

void NeuralNetwork::_train_init() {
  _net_status = STATUS_OKAY;
  _epoch      = 0;
  _loss_prev  = 0.0f;
  _loss       = INF;
  _vloss      = INF;
  _vloss_min  = INF;

  // initialize weights uniformly randomly between [-1, 1]
  for (index_t u = 1; u < _h; ++u) {
    _w[u] = randu<mat>(_k[u], _k[u - 1]) * 2.0f - 1.0f;
    _b[u] = randu<mat>(_k[u], 1) * 2.0f - 1.0f;
    // initialize gradients randomly
    _dw[u] = randu<mat>(_k[u], _k[u - 1]) * 2.0f - 1.0f;
    _db[u] = randu<mat>(_k[u], 1) * 2.0f - 1.0f;
  }
  // initialize deltaw and deltab for rprop and rmsprop
  if (train_algo == TRAIN_RPROP || train_algo == TRAIN_RMSPROP) {
    for (index_t u = 1; u < _h; ++u) {
      _deltaw[u] = mat(_k[u], _k[u - 1]); _deltaw[u].fill(_delta_init);
      _deltab[u] = mat(_k[u], 1);         _deltab[u].fill(_delta_init);
    }
  }

  // check for training_masks if set
  if (!_training_masks.empty()) {
    if (_y.n_cols != _training_masks.size()) {
      string msg = "The size of training examples (" + to_string(_y.n_cols)
        + ") is not equal to the number of masks (" + to_string(_training_masks.size()) + ").";
      error_and_exit(__PRETTY_FUNCTION__, msg);
    }
    for (auto& mask : _training_masks) {
      if (mask.empty()) {
        error_and_exit(__PRETTY_FUNCTION__, "No sampled targets");
      }
      if (mask.max() >= _y.n_rows) {
        error_and_exit(__PRETTY_FUNCTION__, "Some indices of mask is out-of-range");
      }
      mask = arma::sort(mask);
    }
  }
}

void NeuralNetwork::_train_echo(index_t ii) {
  // update all intermediate outputs from hidden layers along the way
  _feedforward(_x.col(ii));

  // compute the last df, which depends on the chosen loss function
  _df.back() = _ell->dloss_df(_f.back(), _y.col(ii));

  // mask out a subset of weights, assume the index in each mask is sorted.
  if (!_training_masks.empty()) {
    index_t j = 0, k = 0;
    while (j < _y.n_rows && k < _training_masks[ii].size()) {
      if (j < _training_masks[ii](k)) {
        _df.back()(j) = 0.0;
      } else {
        k ++;
      }
      j ++;
    }
  }

  // compute the last dg, dw, and db
  _dg.back() = _tfcns.back()->dtransfer(_g.back() + _b.back());
  _dw.back() = (_dg.back() % _df.back().t()) * _f[_h - 2].t();
  _db.back() = _df.back().t() % _dg.back();

  // back propagation to the rest, update df, dg, dw
  for (index_t u = _h - 2; u > 0; --u) {
    _df[u] = (_df[u + 1] % _dg[u + 1].t()) * _w[u + 1];
    _dg[u] = _tfcns[u]->dtransfer(_g[u] + _b[u]);
    _dw[u] = (_dg[u] % _df[u].t()) * _f[u - 1].t();
    _db[u] = _df[u].t() % _dg[u];
  }
}

void NeuralNetwork::_train_sgd() {
  while (!_should_stop()) {
    _epoch ++;
    uvec order = shuffle(regspace<uvec>(0, _n - 1));
    for (index_t ii = 0; ii < _n; ++ii) {
      // feedforward + backpropagate one example
      _train_echo(order(ii));

      // update weights immediately
      for (index_t u = 1; u < _h; ++u) {
        _w[u] -= (eta * _dw[u]);
        _b[u] -= (eta * _db[u]);
      }
    }

#ifdef ENABLE_LOGGING
    print();
#endif

    // update loss
    _loss_prev = _loss;
    _loss = _ell->loss(_feedforward_batch(_x), _y);
  }
}

void NeuralNetwork::_train_gd() {
  // make temporary matrix for holding cummulative gradients over examples
  vector<mat> accu_dw; accu_dw.push_back(zeros<mat>(0, 0));
  vector<mat> accu_db; accu_db.push_back(zeros<mat>(0, 0));
  for (index_t u = 1; u < _h; ++u) { // skip gradients for the 0-th layer
    accu_dw.push_back(zeros<mat>(_k[u], _k[u - 1]));
    accu_db.push_back(zeros<mat>(_k[u], 1));
  }
  while (!_should_stop()) {
    _epoch ++;
    // clear accumulate gradients
    for (index_t u = 1; u < _h; ++u) {
      accu_dw[u].fill(0.0f);
      accu_db[u].fill(0.0f);
    }
    for (index_t ii = 0; ii < _n; ++ii) {
      // feedforward + backpropagate one example
      _train_echo(ii);

      // accumulate gradients
      for (index_t u = 1; u < _h; ++u) {
        accu_dw[u] += _dw[u];
        accu_db[u] += _db[u];
      }
    }
    for (index_t u = 1; u < _h; ++u) {
      _w[u] -= (eta * accu_dw[u]);
      _b[u] -= (eta * accu_db[u]);
    }

#ifdef ENABLE_LOGGING
    print();
#endif

    // update loss
    _loss_prev = _loss;
    _loss = _ell->loss(_feedforward_batch(_x), _y);
  }
}

void NeuralNetwork::_train_mgd() {
  // make temporary matrix for holding cummulative gradients over examples
  vector<mat> accu_dw; accu_dw.push_back(zeros<mat>(0, 0));
  vector<mat> accu_db; accu_db.push_back(zeros<mat>(0, 0));
  for (index_t u = 1; u < _h; ++u) { // skip gradients for the 0-th layer
    accu_dw.push_back(zeros<mat>(_k[u], _k[u - 1]));
    accu_db.push_back(zeros<mat>(_k[u], 1));
  }

  uvec order = shuffle(regspace<uvec>(0, _n - 1)); // an order for traversing data points
  index_t ii = 0; // the index of current data point
  while (true) { // for each minibatch
    if (ii == _n) {
      _epoch ++;

      // update loss
      _loss_prev = _loss;
      _loss = _ell->loss(_feedforward_batch(_x), _y);

      if (_should_stop()) break;
      order = shuffle(order);
      ii = 0; // roll back to the 1st data point for the next epoch
    }
    // clear accumulate gradients
    for (index_t u = 1; u < _h; ++u) {
      accu_dw[u].fill(0.0f);
      accu_db[u].fill(0.0f);
    }
    index_t cnt = 0; // # of data points passed in the current minibatch
    while ((minibatch_size == 0 && ii < _n) || (cnt < minibatch_size)) {
      _train_echo(ii); // "bounce" one example

      // accumulate gradients
      for (index_t u = 1; u < _h; ++u) {
        accu_dw[u] += _dw[u];
        accu_db[u] += _db[u];
      }
      ii ++;
      cnt ++;
    }
    for (index_t u = 1; u < _h; ++u) {
      _w[u] -= (eta * accu_dw[u]);
      _b[u] -= (eta * accu_db[u]);
    }

#ifdef ENABLE_LOGGING
    print();
#endif

  }
}

void NeuralNetwork::_train_rprop() {
  // declear two sets of temporary matrices for holding current/last gradients
  // to avoid copy values from current -> last after every epoch
  // we use two pointers to switch between these two containers.
  vector<mat> dw0; dw0.push_back(zeros<mat>(0, 0));
  vector<mat> db0; db0.push_back(zeros<mat>(0, 0));

  vector<mat> dw1; dw1.push_back(zeros<mat>(0, 0));
  vector<mat> db1; db1.push_back(zeros<mat>(0, 0));

  for (index_t u = 1; u < _h; ++u) { // skip gradients for the 0-th layer
    dw0.push_back(zeros<mat>(_k[u], _k[u - 1]));
    db0.push_back(zeros<mat>(_k[u], 1));

    dw1.push_back(zeros<mat>(_k[u], _k[u - 1]));
    db1.push_back(zeros<mat>(_k[u], 1));
  }

  vector<mat>* curr_dw = &dw0;
  vector<mat>* curr_db = &db0;
  vector<mat>* last_dw = &dw1;
  vector<mat>* last_db = &db1;

  while (!_should_stop()) {
    _epoch ++;
    // clear current gradients
    for (index_t u = 1; u < _h; ++u) {
      (*curr_dw)[u].fill(0.0f);
      (*curr_db)[u].fill(0.0f);
    }
    for (index_t ii = 0; ii < _n; ++ii) {
      _train_echo(ii); // feedforward + backpropagate one example
      // accumulate gradients
      for (index_t u = 1; u < _h; ++u) {
        (*curr_dw)[u] += _dw[u];
        (*curr_db)[u] += _db[u];
      }
    }
    // update weights
    for (index_t u = 1; u < _h; ++u) {
      // update deltaw
      mat mul_w = sign((*curr_dw)[u] % (*last_dw)[u]);
      _deltaw[u].elem( find(mul_w > 0) ) *= _eta_plus;
      _deltaw[u].elem( find(mul_w < 0) ) *= _eta_minus;

      // clamp deltaw
      _deltaw[u] = clamp(_deltaw[u], _delta_min, _delta_max);

      // update deltab
      mat mul_b = sign((*curr_db)[u] % (*last_db)[u]);
      _deltab[u].elem( find(mul_b > 0) ) *= _eta_plus;
      _deltab[u].elem( find(mul_b < 0) ) *= _eta_minus;

      // clamp deltab
      _deltab[u] = clamp(_deltab[u], _delta_min, _delta_max);

      // update weights: w and b
      _w[u] -= (sign((*curr_dw)[u]) % _deltaw[u]);
      _b[u] -= (sign((*curr_db)[u]) % _deltab[u]);
    }
    // swap *curr and *last
    std::swap(curr_dw, last_dw);
    std::swap(curr_db, last_db);

#ifdef ENABLE_LOGGING
    print();
#endif

    // update loss
    _loss_prev = _loss;
    _loss = _ell->loss(_feedforward_batch(_x), _y);
  }
}

void NeuralNetwork::_train_rmsprop() {
  // declear two sets of temporary matrices for holding current/last gradients
  // to avoid copy values from current -> last after every epoch
  // we use two pointers to switch between these two containers.
  vector<mat> dw0; dw0.push_back(zeros<mat>(0, 0));
  vector<mat> db0; db0.push_back(zeros<mat>(0, 0));

  vector<mat> dw1; dw1.push_back(zeros<mat>(0, 0));
  vector<mat> db1; db1.push_back(zeros<mat>(0, 0));

  // these are the moving averages of the squared gradiants for w and b
  vector<mat> Eg2_w; Eg2_w.push_back(zeros<mat>(0, 0));
  vector<mat> Eg2_b; Eg2_b.push_back(zeros<mat>(0, 0));

  for (index_t u = 1; u < _h; ++u) { // skip gradients for the 0-th layer
    dw0.push_back(zeros<mat>(_k[u], _k[u - 1]));
    db0.push_back(zeros<mat>(_k[u], 1));

    dw1.push_back(zeros<mat>(_k[u], _k[u - 1]));
    db1.push_back(zeros<mat>(_k[u], 1));

    Eg2_w.push_back(zeros<mat>(_k[u], _k[u - 1]));
    Eg2_b.push_back(zeros<mat>(_k[u], 1));
  }

  vector<mat>* curr_dw = &dw0;
  vector<mat>* curr_db = &db0;
  vector<mat>* last_dw = &dw1;
  vector<mat>* last_db = &db1;

  uvec order = shuffle(regspace<uvec>(0, _n - 1)); // stores an order for traversing data points
  index_t ii = 0; // the index of current data point

  double beta = 0.9; // moving average parameter in the range [0, 1], should be closer to 1
  bool not_set = true;

  while (true) {
    if (ii == _n) {
      _epoch ++;

#ifdef ENABLE_LOGGING
    print();
#endif

      // update loss
      _loss_prev = _loss;
      _loss = _ell->loss(_feedforward_batch(_x), _y);

      if (_should_stop()) break;
      order = shuffle(order);
      ii = 0; // roll back to the first data point for the next epoch
    }
    // clear current gradients
    for (index_t u = 1; u < _h; ++u) {
      (*curr_dw)[u].fill(0.0f);
      (*curr_db)[u].fill(0.0f);
    }
    index_t cnt = 0; // # of data points traversed in the current minibatch
    while ((minibatch_size == 0 && ii < _n) || (cnt < minibatch_size)) {
      _train_echo(order(ii));  // feedforward + backpropagate one example
      // accumulate gradients
      for (index_t u = 1; u < _h; ++u) {
        (*curr_dw)[u] += _dw[u];
        (*curr_db)[u] += _db[u];
      }
      ii ++;
      cnt ++;
    }

    if (not_set) {
      beta = 0;
      not_set = false;
    } else {
      beta = 0.9;
    }

    // update weights using gradients from this minibatch
    for (index_t u = 1; u < _h; ++u) {
      // update Eg2_w, Eg2_b
      Eg2_w[u] = beta * Eg2_w[u] + (1 - beta) * arma::square((*curr_dw)[u]);
      Eg2_b[u] = beta * Eg2_b[u] + (1 - beta) * arma::square((*curr_db)[u]);

      // update deltaw
      mat mul_w = sign((*curr_dw)[u] % (*last_dw)[u]);
      _deltaw[u].elem( find(mul_w > 0) ) *= _eta_plus;
      _deltaw[u].elem( find(mul_w < 0) ) *= _eta_minus;

      // clamp deltaw
      _deltaw[u] = clamp(_deltaw[u], _delta_min, _delta_max);

      // update deltab
      mat mul_b = sign((*curr_db)[u] % (*last_db)[u]);
      _deltab[u].elem( find(mul_b > 0) ) *= _eta_plus;
      _deltab[u].elem( find(mul_b < 0) ) *= _eta_minus;

      // clamp deltab
      _deltab[u] = clamp(_deltab[u], _delta_min, _delta_max);

      // update weights: w and b
      // _w[u] -= ((*curr_dw)[u] % _deltaw[u] / arma::sqrt(Eg2_w[u]));
      // _b[u] -= ((*curr_db)[u] % _deltab[u] / arma::sqrt(Eg2_b[u]));

      // update weights: w and b using fixed learning rate
      _w[u] -= ((*curr_dw)[u] * eta / arma::sqrt(Eg2_w[u]));
      _b[u] -= ((*curr_db)[u] * eta / arma::sqrt(Eg2_b[u]));
    }
    // swap *curr and *last
    std::swap(curr_dw, last_dw);
    std::swap(curr_db, last_db);
  }
}

bool NeuralNetwork::_should_stop() {
  if (_epoch >= epochs) {
    _net_status = STATUS_REACHED_MAX_EPOCH;
    return true;
  }
  if (_loss <= goal) {
    _net_status = STATUS_REACHED_GOAL;
    return true;
  }
  if (early_stopping && _vn > 0) {
    // evaluate on the validation partition
    _vloss = _ell->loss(_feedforward_batch(_vx), _vy);
    if (_vloss > _vloss_min) {
      _fails += 1;
    } else { // update the optimal model
      _optimalw  = _w;
      _optimalb  = _b;
      _vloss_min = _vloss;
      _fails     = 0; // reset # of fails
    }
    if (_fails >= max_fails) {
      _net_status = STATUS_REACHED_MAX_FAIL;
      return true;
    }
  }
  double grad2 = 0.0; // squared magnitude of gradient: \nebla(loss)^2
  for (index_t u = 1; u < _h; ++u) {
    grad2 += arma::accu(arma::square(_dw[u]));
    grad2 += arma::accu(arma::square(_db[u]));
  }
  if (std::sqrt(grad2) <= min_grad || std::fabs(_loss - _loss_prev) <= min_diff) {
    _net_status = STATUS_CONVERGED;
    return true;
  }
  _net_status = STATUS_OKAY;
  return false;
}

void NeuralNetwork::_feedforward(const mat& x) {
  _f[0] = x;
  for (index_t u = 1; u < _h; ++u) {
    _g[u] = _w[u] * _f[u - 1];
    _f[u] = _tfcns[u]->transfer(_g[u] + _b[u]);
  }
}

mat NeuralNetwork::_feedforward_batch(const mat& x) const {
  mat gu, f(x);
  for (index_t u = 1; u < _h; ++u) {
    gu = _w[u] * f;
    gu.each_col() += _b[u]; // for each data point, in-place addition of bias
    f = _tfcns[u]->transfer(gu);
  }
  return f;
}

void NeuralNetwork::serialize(std::ostream& stream) const {
  uint32_t data;
  data = _h;
  SAVE_VAR(stream, data);
  // save the structure of the network
  for (index_t u = 0; u < _h; ++u) {
    data = _k[u];
    SAVE_VAR(stream, data);
  }
  for (index_t u = 1; u < _h; ++u) {
    _w[u].save(stream);
    _b[u].save(stream);
    data = _tfcns_id[u];
    SAVE_VAR(stream, data);
  }
  data = _ell_id;
  SAVE_VAR(stream, data);
}

void NeuralNetwork::deserialize(std::istream& stream) {
  uint32_t n, data;
  LOAD_N(stream, n, sizeof(uint32_t)); // n: number of layers

  LOAD_N(stream, data, sizeof(uint32_t)); // size of the first layer (input)
  index_t input_size = data;

  // load the structure of the network
  vector<index_t> layers;
  for (uint32_t u = 0; u < n - 2; ++u) {
    LOAD_N(stream, data, sizeof(uint32_t));
    layers.push_back(data);
  }

  LOAD_N(stream, data, sizeof(uint32_t)); // size of the last layer (output)
  index_t output_size = data;

  // Note that resize calls _init() which initializes the network to a fresh state with a fixed
  // structure, other customization are set afterwards.
  resize(layers);

  _k.front() = input_size;
  _k.back()  = output_size;
  for (index_t u = 1; u < _h; ++u) {
    _w[u].load(stream);
    _b[u].load(stream);
    LOAD_N(stream, data, sizeof(uint32_t));
    set_transfer_function(u, data);
  }
  LOAD_N(stream, data, sizeof(uint32_t));
  // for some loss functions, this only sets a placeholder, the additional parameters will be set
  // during training.
  set_loss_function(data);
}

const NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork &other) {
  Model::operator=(other);
  if (this != &other) {
    // a deep copy of all data
    train_algo     = other.train_algo;
    early_stopping = other.early_stopping;
    eta            = other.eta;
    min_diff       = other.min_diff;
    min_grad       = other.min_grad;
    max_fails      = other.max_fails;
    goal           = other.goal;
    epochs         = other.epochs;
    minibatch_size = other.minibatch_size;

    vector<index_t> hidden_layers;
    for (index_t u = 1; u < other._h - 1; ++u) {
      hidden_layers.push_back(other._k[u]);
    }
    resize(hidden_layers);

    // copy the first (input) and last (output) dimension
    _k.front() = other._k.front();
    _k.back()  = other._k.back();

    _n  = other._n;
    _x  = other._x;
    _y  = other._y;
    _vp = other._vp;
    _vn = other._vn;
    _vx = other._vx;
    _vy = other._vy;

    for (index_t u = 0; u < _h; ++u) {
      _f[u]  = other._f[u];
      _w[u]  = other._w[u];
      _g[u]  = other._g[u];
      _b[u]  = other._b[u];
      _df[u] = other._df[u];
      _dw[u] = other._dw[u];
      _dg[u] = other._dg[u];
      _db[u] = other._db[u];

      _optimalw[u] = other._optimalw[u];
      _optimalb[u] = other._optimalb[u];

      _deltaw[u] = other._deltaw[u];
      _deltab[u] = other._deltab[u];

      _tfcns_id[u] = other._tfcns_id[u];
      _tfcns[u]    = other._tfcns[u];
    }

    _eta_plus   = other._eta_plus;
    _eta_minus  = other._eta_minus;
    _delta_max  = other._delta_max;
    _delta_min  = other._delta_min;
    _delta_init = other._delta_init;

    _epoch = other._epoch;

    _loss      = other._loss;
    _loss_prev = other._loss_prev;

    _fails = other._fails;
    _vloss = other._vloss;
    _vloss_min = other._vloss_min;

    _net_status = other._net_status;

    _ell_id = other._ell_id;
    _ell    = other._ell;
  }
  return *this;
}

// -------------
// Yuxiang Jiang (yuxjiang@indiana.edu)
// Department of Computer Science
// Indiana University Bloomington
// Last modified: Tue 17 Sep 2019 11:50:55 PM P
