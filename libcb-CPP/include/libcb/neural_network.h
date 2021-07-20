#ifndef _LIBCB_NEURAL_NETWORK_H_
#define _LIBCB_NEURAL_NETWORK_H_

#include <memory> // shared_ptr()
#include <cmath>

#include "util.h"
#include "classification.h"

//////////////////////
// BASE: Loss function
//////////////////////
/**
 * @brief abstract class of loss functions
 *
 * @remark all user-defined loss functions should inherit from this class and implement the two
 * virtual "interface" methods: loss and dloss_df.
 */
struct LossFcn {
  std::string name;
  LossFcn(std::string tag = "undefined") : name(tag) {}
  virtual ~LossFcn() {}

  /**
   * @brief compute loss
   *
   * @param f the prediction matrix, one column corresp. to one example
   * @param y the ground truth matrix for each data points, should be of the same size as f.
   */
  virtual double loss(const arma::mat& f, const arma::mat& y) const = 0;

  /**
   * @brief compute gradient, dloss/df, where f is the output of the whole network
   *
   * @remark Although this function takes care of the general case where both inputs are matrices
   * representing a batch of training examples, yet this function is currently only called by
   * _train_echo() which "echos" one single example, because this reduces the memory footprint.
   *
   * @param f the prediction matrix, one column corresp. to one example
   * @param y the ground truth matrix for each data points, should be of the same size as f.
   */
  virtual arma::mat dloss_df(const arma::mat& f, const arma::mat& y) const = 0;

  /**
   * @brief transform the output f to be a consistent output
   *
   * @remark for loss functions that doesn't have this consistency features, simply implement an
   * identity transform should be enough, as it shouldn't be called in those cases anyway.
   */
  virtual arma::mat consistent_output(const arma::mat& f) const = 0;
};

/**
 * @brief the mean squared error function class.
 *
 * @remark MSE(f, y) = 1/(2m) (f - y)^T (f - y), where f is the prediction, y is the ground truth
 * and m is the number of neurons.
 */
struct MSE : public LossFcn {
  MSE() : LossFcn("mean squared error") {}

  double loss(const arma::mat& f, const arma::mat& y) const {
    // note that the squared loss is averaged over all neurons and all examples (if more than one
    // example is presented)
    return 0.5f * arma::accu(arma::square(f - y)) / f.n_elem;
  }

  arma::mat dloss_df(const arma::mat& f, const arma::mat& y) const {
    // note that f.n_rows = m, which is the number of neurons in the output layer
    return (f - y).t() / f.n_rows;
  }

  arma::mat consistent_output(const arma::mat& f) const {
    return f;
  }
};

/**
 * @brief the mean absolute error function class.
 *
 * @remark MAE(f, y) = 1/(2m) |f - y|
 */
struct MAE : public LossFcn {
  MAE() : LossFcn("mean absolute error") {}

  double loss(const arma::mat& f, const arma::mat& y) const {
    return arma::accu(arma::abs(f - y)) / f.n_elem;
  }

  arma::mat dloss_df(const arma::mat& f, const arma::mat& y) const {
    return arma::sign(f - y).t() / f.n_rows;
  }

  arma::mat consistent_output(const arma::mat& f) const {
    return f;
  }
};

/**
 * @brief the cross-entropy error function class.
 */
struct CrossEntropy : public LossFcn {
  CrossEntropy() : LossFcn("cross entropy") {}

  double loss(const arma::mat& f, const arma::mat& y) const {
    return arma::accu(y % arma::log(arma::clamp(y, EPS, 1.0f)) +
        (1.0f - y) % arma::log(arma::clamp(1.0f - y, EPS, 1.0f)));
  }

  arma::mat dloss_df(const arma::mat& f, const arma::mat& y) const {
    arma::mat clamped_f = arma::clamp(f, 0.0f, 1.0f);
    arma::mat clamped_y = arma::clamp(y, 0.0f, 1.0f);
    return (-clamped_y / (clamped_f + EPS) + (1.0f - clamped_y) / (1.0f - clamped_f + EPS)).t();
  }

  arma::mat consistent_output(const arma::mat& f) const {
    return f;
  }
};

/**
 * @brief the mean squared error function class for ConFANNGO.
 */
struct MSECon : public LossFcn {
  arma::mat A;

  MSECon(const arma::mat& A_) : LossFcn("consistent mean squared error") {
    if (!A_.is_square()) {
      error_and_exit(__PRETTY_FUNCTION__, "Ancestor matrix must be square.");
    }
    A = A_;
  }

  double loss(const arma::mat& f, const arma::mat& y) const {
    // averaged over all neurons and examples
    return 0.5f * arma::accu(arma::square(consistent_output(f) - y)) / f.n_elem;
  }

  arma::mat dloss_df(const arma::mat& f, const arma::mat& y) const {
    // dloss/df = dloss/dg * dg/df
    return ((consistent_output(f) - y).t() / f.n_rows) * A / f.n_rows;
  }

  arma::mat consistent_output(const arma::mat& f) const {
    return (A * f) / f.n_rows;
  }
};

/**
 * @brief the semantic distance loss funcition for ConFANNGO-SD
 */
struct SD : public LossFcn {
  arma::mat A;

  SD(const arma::mat& A_) : LossFcn("semantic distance") {
    if (!A.is_square()) {
      error_and_exit(__PRETTY_FUNCTION__, "Ancestor matrix must be square.");
    }
    A = A_;
  }

  double loss(const arma::mat& f, const arma::mat& y) const {
    index_t m = f.n_rows; // number of neurons
    index_t n = f.n_cols; // number of data points
    arma::mat g    = (A * f) / m;
    arma::mat diff = g - y;

    double d = 0.0;
    for (index_t i = 0; i < n; ++i) {
      double pos(0.0), neg(0.0);
      for (const auto& value : diff.col(i)) {
        if (value > 0) {
          pos += value;
        } else {
          neg += value;
        }
      }
      d += ::sqrt(pos * pos + neg * neg);
    }
    return d / static_cast<double>(n);
  }

  arma::mat dloss_df(const arma::mat& f, const arma::mat& y) const {
    double d = 0.0;
    index_t m = f.n_rows;
    index_t n = f.n_cols;

    arma::mat g = (A * f) / m;

    arma::mat partial(m, n, arma::fill::zeros);
    for (index_t i = 0; i < n; ++i) {
      double pos(0.0), neg(0.0);
      for (index_t j = 0; j < m; ++j) {
        if (g(j, i) > y(j, i)) {
          pos += g(j, i) - y(j, i);
        } else {
          neg += y(j, i) - g(j, i);
        }
      }
      // second round
      for (index_t j = 0; j < m; ++j) {
        if (g(j, i) > y(j, i)) {
          partial(j, i) = pos;
        } else if (g(j, i) < y(j, i)) {
          partial(j, i) = neg;
        } else {
          // this is ill-posed, as it is not differentiable
          // but should be fine in practice
          partial(j, i) = 0;
        }
      }
      // update d
      d += ::sqrt(pos * pos + neg * neg);
    }
    d /= static_cast<double>(n); // normalize by data point counts
    return partial.t() * A / m / d;
  }

  arma::mat consistent_output(const arma::mat& f) const {
    return A * f; // TODO
  }
};

/**
 * @brief the mean squared error function class for ConFANNGO.
 */
struct MSEConReg : public LossFcn {
  arma::mat A;
  arma::mat ATA; // A^{T} * A

  MSEConReg(const arma::mat& A_) : LossFcn("consistent mean squared error for regression") {
    if (!A_.is_square()) {
      error_and_exit(__PRETTY_FUNCTION__, "Ancestor matrix must be square.");
    }
    A = A_;
    ATA = A_.t() * A_;
  }

  double loss(const arma::mat& f, const arma::mat& y) const {
    // averaged over all neurons and examples
    return 0.5f * arma::accu(arma::square(consistent_output(f - y))) / f.n_elem;
  }

  arma::mat dloss_df(const arma::mat& f, const arma::mat& y) const {
    // dloss/df = dloss/dg * dg/df
    return (f - y).t() * ATA / f.n_rows; // regression
  }

  arma::mat consistent_output(const arma::mat& f) const {
    return A * f; // regression
  }
};

///////////////////////////////////////
// BASE: Transfer (activation) function
///////////////////////////////////////
/**
 * @brief abstract class of transfer functions
 *
 * @remark all user-defined transfer functions should inherit from this class and implement the two
 * virtual "interface" methods: transfer and gradient.
 */
struct TransFcn {
  std::string name;
  TransFcn(std::string tag = "undefined") : name(tag) {}
  virtual ~TransFcn() {}

  // transfer
  virtual arma::mat transfer(const arma::mat& x) const = 0;

  // gradient
  virtual arma::mat dtransfer(const arma::mat& x) const = 0;
};

/**
 * @brief the sigmoid transfer function class.
 */
struct Sigmoid : public TransFcn {
  Sigmoid() : TransFcn("sigmoid") {}
  arma::mat transfer(const arma::mat& x) const {
    return 1.0f / (1.0f + arma::exp(-x));
  }
  arma::mat dtransfer(const arma::mat& x) const {
    arma::mat sigmoid_x = 1.0f / (1.0f + arma::exp(-x));
    return sigmoid_x % (1.0f - sigmoid_x);
  }
};

/**
 * @brief the tanh transfer function class.
 */
struct Tanh : public TransFcn {
  Tanh() { name = std::string("tanh"); }
  arma::mat transfer(const arma::mat& x) const {
    return arma::tanh(x);
  }
  arma::mat dtransfer(const arma::mat& x) const {
    return 1.0f / arma::square(arma::cosh(x));
  }
};

/**
 * @brief the tansig transfer function class.
 */
struct Tansig : public TransFcn {
  Tansig() : TransFcn("tansig") {}
  arma::mat transfer(const arma::mat& x) const {
    return 2.0f / (1.0f + arma::exp(-2.0f * x)) - 1.0f;
  }
  arma::mat dtransfer(const arma::mat& x) const {
    arma::mat sigmoid_x = 1.0f / (1.0f + arma::exp(-2.0f * x));
    return (sigmoid_x % (1.0f - sigmoid_x)) * 4.0f;
  }
};

/**
 * @brief the purelin transfer function class.
 */
struct Purelin : public TransFcn {
  double k = 1.0f; // identity function
  Purelin() : TransFcn("purelin") { k = 1.0f; }
  arma::mat transfer(const arma::mat& x) const {
    return x;
  }
  arma::mat dtransfer(const arma::mat& x) const {
    return arma::ones<arma::mat>(x.n_rows, 1);
  }
};

/**
 * @brief the neural network class.
 */
class NeuralNetwork : public Model {
  public:
    //! network status
    enum Status {
      STATUS_OKAY,              // running
      STATUS_ERROR,             // error
      STATUS_CONVERGED,         // converged
      STATUS_REACHED_GOAL,      // reached minimum loss
      STATUS_REACHED_MAX_EPOCH, // maximum epoch number reached
      STATUS_REACHED_MAX_FAIL   // maximum fail number reached
    };

    //! training algorithms
    enum Algorithm {
      /**
       * gradient descent has several variants depending on the minibatch size
       * minibatch size = 0 or >= training size (n) --> batch mode (gd)
       * minibatch size = 1 --> stochastic gradient descent (sgd)
       * otherwise --> minibatch mode (mgd)
       */
      TRAIN_GD,

      /**
       * resilient propagation
       */
      TRAIN_RPROP,  

      /**
       * root mean square propagation (rmsprop) is the "minibatch" version of rprop
       */
      TRAIN_RMSPROP
    };

    /** Registered loss function
     * @remark that any derived loss function should be registered here in order to save the entire
     * network to file, although loss function is not necessary for prediction
     */
    static const index_t LS_MSE          = 0;
    static const index_t LS_MAE          = 1;
    static const index_t LS_CROSSENTROPY = 2;
    static const index_t LS_SD           = 3;
    // extensions ...
    // static const index_t LS_XXX = N;
    static const index_t LS_MSECON       = 4; // consistent version of MSE
    static const index_t LS_MSECONREG    = 5; // consistent and regression version of MSE

    /** Registered transfer function
     * @remark that any derived transfer function should be registered here in order to save the
     * entire network to file, since one has to store the type of transfer function so as to restore
     * a network model.
     */
    static const index_t TF_SIGMOID = 0;
    static const index_t TF_TANH    = 1;
    static const index_t TF_TANSIG  = 2;
    static const index_t TF_PURELIN = 3;
    // extensions ...
    // const index_t TF_XXX = N;

  public:
    /**
     * @brief the default constructor.
     *
     * @remark it constructs a network with 1 hidden layer.
     */
    NeuralNetwork() {
      resize({10});
    }

    NeuralNetwork(index_t n_hidden) {
      resize({n_hidden});
    }

    /**
     * @brief the constructor for more general networks (potentially deep)
     *
     * @param layers number of units/neurons in each hidden layer, starting from the input layer to
     * the output layer.
     *
     * @remark number of layers must be at least 3.
     */
    NeuralNetwork(const std::vector<index_t>& hidden_layers) {
      resize(hidden_layers);
    }

    /**
     * @brief resizes the network dimensions (deep network version).
     *
     * @remark resizing the network causes all training (and validation as well if applicable) data
     * to be discarded.
     *
     * @param layers number of units/neurons in each layer, starting from the input layer to the
     * output layer.
     *
     * @remark number of layers must be at least 3.
     */
    void resize(const std::vector<index_t>& hidden_layers);

    /**
     * @brief set loss function for this network.
     *
     * @param fcn the function.
     * @param params the additional parameters needed for the initializion of some specific loss
     * functions
     */
    inline void set_loss_function(index_t id,
        const std::vector<arma::mat>& params = std::vector<arma::mat>()) {
      switch (id) {
        case LS_MAE:
          _ell = std::make_shared<MAE>(MAE());
          break;
        case LS_CROSSENTROPY:
          _ell = std::make_shared<CrossEntropy>(CrossEntropy());
          break;
        case LS_MSECON:
          if (!params.empty()) {
            _ell = std::make_shared<MSECon>(MSECon(params[0]));
          } else {
            // set the ancestor matrix A as a dummy place holder
            _ell = std::make_shared<MSECon>(MSECon(arma::zeros(1)));
          }
          break;
        case LS_MSECONREG:
          if (!params.empty()) {
            _ell = std::make_shared<MSEConReg>(MSEConReg(params[0]));
          } else {
            // set the ancestor matrix A as a dummy place holder
            _ell = std::make_shared<MSEConReg>(MSEConReg(arma::zeros(1)));
          }
          break;
        case LS_SD:
          if (!params.empty()) {
            _ell = std::make_shared<SD>(SD(params[0]));
          } else {
            // set the ancestor matrix A as a dummy place holder
            _ell = std::make_shared<SD>(SD(arma::zeros(1)));
          }
          break;
        case LS_MSE:
        default:
          _ell = std::make_shared<MSE>(MSE());
          id = LS_MSE;
          break;
      }
      _ell_id = id;
    }

    inline index_t get_loss_function() const {
      return _ell_id;
    }

    /**
     * @brief set transfer (activation) function for a layer.
     *
     * @param u layer index.
     * @param id the corresponding transfer function id.
     */
    inline void set_transfer_function(index_t u, index_t id) {
      switch (id) {
        case TF_TANH:
          _tfcns[u] = std::make_shared<Tanh>(Tanh());
          break;
        case TF_TANSIG:
          _tfcns[u] = std::make_shared<Tansig>(Tansig());
          break;
        case TF_PURELIN:
          _tfcns[u] = std::make_shared<Purelin>(Purelin());
          break;
        case TF_SIGMOID:
        default:
          _tfcns[u] = std::make_shared<Sigmoid>(Sigmoid());
          id = TF_SIGMOID; // force the default id
          break;
      }
      _tfcns_id[u] = id;
    }

    /**
     * @brief trains this neural network.
     *
     * @remark this is a portal function in that specific training algorithms will be called within
     * this function accordingly.
     *
     * @remark this function should run after (i) pouring the training data and (ii) initializing
     * all parameters or accepting all default parameters.
     */
    void train(const arma::mat& x, const arma::mat& y);

    /**
     * @brief save as train but overwrite the validation proportion with the given parameter.
     */
    void train_with_validation(const arma::mat& x, const arma::mat& y, double p) {
      _vp = p;
      train(x, y);
    }

    /**
     * @brief predicts a data set (feature matrix)
     *
     * @param x the feature matrix, with dimension [n x _k[0]]
     *
     * @return a scoring matrix, with dimension [n x _k[h-1]]
     */
    arma::mat predict(const arma::mat& x) const; // predicts on a data matrix

    /**
     * @brief sets aside a proportion of data points for validation purposes.
     *
     * @param p the validation proportion, must be within the interval [0, 1).
     */
    void reserve_validation_proportion(double p) {
      if (p < 0 || p >= 1) {
        error_and_exit(__PRETTY_FUNCTION__, "The proportion must be in [0, 1)");
      }
      _vp = p;
    }

    void print() const;

    void print_debug() const;

    /**
     * @brief serialize to an output stream.
     *
     * @param out an output stream.
     */
    void serialize(std::ostream& out) const;

    /**
     * @brief deserialize from an input stream.
     *
     * @param in an input stream.
     */
    void deserialize(std::istream& in);

    /**
     * @brief copy from another model and overwrite itself.
     *
     * @param other another neural network.
     */
    const NeuralNetwork& operator=(const NeuralNetwork& other);

  protected:
    /**
     * @brief initialize the size of the neural network and set default parameters.
     *
     * @remark this function is called everytime right after the network is resized.
     */
    void _init();

    /**
     * @brief set data for this neural network.
     *
     * @remark this function doesn't change the structure of hidden layer(s) of the network.
     *
     * @remark the attached data will be splitted into trainging + validation according to _vp
     * before train().
     *
     * @param x the data matrix x.
     * @param y the label matrix y;
     */
    void _attach_data(const arma::mat& x, const arma::mat& y);

    void _make_validation();

    //! initialize weights
    void _train_init();

    /**
     * @brief feedforward and backpropagate one example.
     *
     * @remark this function updates the following internal matrices: f, g (in the forward pass) and
     * df, dg, dw, db (in the backward pass).
     *
     * @param ii is the example index.
     */
    void _train_echo(index_t ii);

    //! training algorithms
    void _train_gd();      // batch gradient descent
    void _train_mgd();     // mini-batch gradient descent
    void _train_sgd();     // stochastic gradient descent
    void _train_rprop();   // resilient backpropagation
    void _train_rmsprop(); // root mean square propagation
    void _train_gdm();     // TODO gradient descent with momentum

    /**
     * @brief feedforward a single training example.
     *
     * @param x a training example as an Armadillo column vector.
     */
    void _feedforward(const arma::mat& x);

    /**
     * @brief feedforward a batch of examples.
     *
     * @remark this function is basically the protected version of predict(), except x and y (the
     * output) is not transposed, that is each example corresponds to a column of x (and y).
     *
     * @param x a batch of examples.
     *
     * @return a transposed scoring matrix.
     */
    arma::mat _feedforward_batch(const arma::mat& x) const;

    /**
     * @brief check if training should stop.
     *
     * @return True or false.
     */
    bool _should_stop();

  protected:
    index_t _h; // number of layers; including input/output

    /** [IMPORTANT NOTE]
     * For the convenience of matrix operations, the NeuralNetwork class stores user data (training
     * + validation) in a transposed manner, i.e., each data point (example) corresponds to a column
     * in _x and _y.
     */
    index_t   _n; // number of training examples
    arma::mat _x; // [m x n], training x
    arma::mat _y; // [p x n], training y

    double    _vp; // the validation proportion
    index_t   _vn; // number of validation examples
    arma::mat _vx; // [m x vn], validation x
    arma::mat _vy; // [p x vn], validation y

    //! internal matrices
    std::vector<index_t>   _k;  // number of neurons of each layer
    std::vector<arma::mat> _f;  // output from each layer
    std::vector<arma::mat> _w;  // weights to each layer from the one before
    std::vector<arma::mat> _g;  // weighted summation from the previous layer
    std::vector<arma::mat> _b;  // bias of each layer
    std::vector<arma::mat> _df; // partial(l) / partial(f) (row vector)
    std::vector<arma::mat> _dg; // partial(f) / partial(g) (diagonal as vector)
    std::vector<arma::mat> _dw; // partial(l) / partial(w) (matrix)
    std::vector<arma::mat> _db; // partial(l) / partial(b) (column vector)

    //! optimal network parameter set
    std::vector<arma::mat> _optimalw;
    std::vector<arma::mat> _optimalb;

    //! parameters for resilient backpropagation
    std::vector<arma::mat> _deltaw; // step size for updating w
    std::vector<arma::mat> _deltab; // step size for updating b

    double _eta_plus   = 1.2f; // step expanding factor
    double _eta_minus  = 0.5f; // step shrinking factor
    double _delta_max  = 5e+2;
    double _delta_min  = 1e-6;
    double _delta_init = 1e-1;

    //! number of epoch (iteration index)
    index_t _epoch;

    double _loss;      // loss over training examples (current iteration)
    double _loss_prev; // loss over training examples (previous iteration)

    //! early stopping
    index_t _fails;     // number of fails (larger validation loss)
    double  _vloss;     // loss over valiadation examples (current iteration)
    double  _vloss_min; // minimum validation loss ever

    //! network parameter
    Status _net_status;

    //! network function pointers
    index_t _ell_id; // registered loss function id, paired with pointers
    std::shared_ptr<LossFcn> _ell;  // loss function pointer
    std::vector<index_t> _tfcns_id; // registered transfer function id, paired with pointers
    std::vector<std::shared_ptr<TransFcn>> _tfcns;  // transfer function pointers

    ////////////////////
    // Runtime variables
    ////////////////////

    /**
     * training_masks is a collection of indices which chooses a subset of the last weight matrix to
     * be updated for each training example. By default, weights associated with all output is
     * updated, however, in cases when (negative) sampling is turned on, weights are updated
     * partially.
     *
     * @remark each mask consists of a list of indices of selected target.
     */
    std::vector<arma::uvec> _training_masks;

  public:
    //! training algorighm
    Algorithm train_algo;

    bool early_stopping; // to apply early stopping stratergy or not

    double  eta;            // learning rate
    double  min_diff;       // minimum difference in the loss function
    double  min_grad;       // minimum gradient magnitude
    index_t max_fails;      // maximum number of validation increases
    double  goal;           // minimum performance value
    index_t epochs;         // maximum number of training epochs
    index_t minibatch_size; // size of minibatches. Special cases, 1: stochastic; 0: full batch
};

#endif // _LIBCB_NEURAL_NETWORK_H_

// -------------
// Yuxiang Jiang (yuxjiang@indiana.edu)
// Department of Computer Science
// Indiana University Bloomington
// Last modified: Tue 17 Sep 2019 09:37:55 PM P
