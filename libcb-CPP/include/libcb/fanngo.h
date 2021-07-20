#ifndef _LIBCB_FANNGO_H_
#define _LIBCB_FANNGO_H_

#include <iostream>
#include <vector>
#include <string>
#include <armadillo>

#include "util.h"
#include "neural_network.h"
#include "classification.h"
#include "ontology.h"

struct FANNGOParam {
  // hyper parameters
  index_t    max_sequence; // 0: no limit on training samples
  index_t    max_output;   // 0: no limit on output

  bool       do_training_subsampling;

  bool       do_feature_selection;
  arma::uvec selected_features;

  bool       do_normalization;
  arma::mat  mus; // both mu and signa are 1-by-n row vectors
  arma::mat  sigmas;

  bool       do_dimension_reduction;
  double     ret_ratio;
  arma::mat  coeff;

  FANNGOParam () {
    max_sequence            = 0;
    max_output              = 0;
    do_training_subsampling = false;
    do_feature_selection    = false;
    do_normalization        = true;
    do_dimension_reduction  = false;
    ret_ratio               = 0.99;
  }

  void print() const;

  const FANNGOParam& operator=(const FANNGOParam& other) {
    if (this != &other) {
      max_sequence            = other.max_sequence;
      max_output              = other.max_output;
      do_training_subsampling = other.do_training_subsampling;
      do_feature_selection    = other.do_feature_selection;
      selected_features       = other.selected_features;
      do_normalization        = other.do_normalization;
      mus                     = other.mus;
      sigmas                  = other.sigmas;
      do_dimension_reduction  = other.do_dimension_reduction;
      ret_ratio               = other.ret_ratio;
      coeff                   = other.coeff;
    }
    return *this;
  }

  void serialize(std::ostream& stream) const;

  void deserialize(std::istream& stream);
};

class FANNGOSingle : public NeuralNetwork {
  public:
    FANNGOParam param;
    Ontology ontology;

  public:
    FANNGOSingle() : NeuralNetwork() {}

    /**
     * trains a single (Con)FANNGO model
     *
     * @param x is the feature matrix
     * @param y is an ordered target matrix
     *
     * @remark the caller of this function have to make sure that the column order of y must align
     * with the topological order of the attach ontology, which is returned by Ontology::terms()
     */
    void train(const arma::mat& x, const arma::mat& y);

    /**
     * predicts on a feature matrix using a trained (Con)FANNGO model
     *
     * @param x is the feature matrix
     * @return an ordered target matrix, where the ordered columns corresponds to terms return be
     * Ontology::terms().
     */
    arma::mat predict(const arma::mat& x) const;

    const FANNGOSingle& operator=(const FANNGOSingle& other) {
      NeuralNetwork::operator=(other);
      if (this != &other) {
        param    = other.param;
        ontology = other.ontology;
      }
      return *this;
    }

    bool is_consistent() const {
      index_t fcn_id = get_loss_function();
      switch (fcn_id) {
        case NeuralNetwork::LS_MSECON:
        case NeuralNetwork::LS_MSECONREG:
        case NeuralNetwork::LS_SD:
          return true;
        default:
          return false;
      }
      return false;
    }

    void serialize(std::ostream& stream) const {
      NeuralNetwork::serialize(stream);
      param.serialize(stream);
      ontology.serialize(stream);
    }

    void deserialize(std::istream& stream) {
      NeuralNetwork::deserialize(stream);
      param.deserialize(stream);
      ontology.deserialize(stream);
      // Note: since we serialize the ID of the loss function, which will be used to determine if
      // the model is consistent or not and we can supply parameters (the A() matrix) needed to do
      // prediction
      if (is_consistent()) {
        std::vector<arma::mat> params;
        params.push_back(ontology.A());
        set_loss_function(get_loss_function(), params);
      }
    }

  protected:
    /**
     * @brief builds/sets a subset of target index for each example to update during training
     */
    void _prepare_ont_training(const arma::mat& y);

    /**
     * @brief select a subset of features for i-th network in the ensemble.
     *
     * @remark this function modifies m_selected_features.
     *
     * @param i the index of network.
     * @param x the feature matrix (subject to modification).
     * @param y the labelling matrix.
     */
    void _feature_selection(arma::mat& x, const arma::mat& y);

    /**
     * @brief normalize the feature matrix for i-th network in the ensemble.
     *
     * @remark this function modifies m_mus and m_sigmas.
     *
     * @param i the index of network.
     * @param x the feature matrix (subject to modification).
     */
    void _normalization(arma::mat& x);

    /**
     * @brief reduce the dimensionality of the feature space using pca.
     *
     * @remark this function modifies m_coeffs.
     *
     * @param i the index of network.
     * @param x the feature matrix (subject to modification).
     * @param ret the retained variance.
     */
    void _dimension_reduction(arma::mat& x);
};

class FANNGO : public Ensemble<FANNGOSingle> {
  public:
    FANNGOParam param;
    Ontology ontology;

  public:
    FANNGO() {}

    FANNGO(index_t n, const Ontology& o, const FANNGOParam& p) : Ensemble<FANNGOSingle>(n) {
      ontology = o;
      param    = p;
    }

    /**
     * @brief Prepares the training data for the i-th model:
     * + make a bootstrap sample
     *   -> update: sampled_examples
     * + (sub-)sample output terms if necessary
     *   -> update: sample_targets
     * + create an corresponding (sub-sampled) ontology for the i-th model.
     *   -> update: ontology
     *
     * @param i the index of models.
     * @param x the feature matrix.
     * @param y the label matrix.
     *
     * @remark the order of sampled_targets must align with the topological order of (sub-sampled)
     * ontology for i-th model (i.e. returned by Ontology::terms() of that ontology)
     */
    void train_setup_i(index_t i, const arma::mat& x, const arma::mat& y);

    /**
     * @brief Combines with another fanngo model
     *
     * @param other, another fanngo model which has the same attached ontology.
     */
    void combine(const FANNGO& other);

    /**
     * @remark the caller of train(x, y) has to make sure that the column order of y must align with
     * the topological order returned by Ontology::terms()
     */

    /**
     * @override Ensemble::predict() function to utilize the ontology structure underlies the output
     * layer.
     */
    arma::mat predict(const arma::mat& x) const;

    const FANNGO& operator=(const FANNGO& other) {
      Ensemble<FANNGOSingle>::operator=(other);
      if (this != &other) {
        param    = other.param;
        ontology = other.ontology;
      }
      return *this;
    }

    /**
     * @brief Saves a network ensemble model to a file.
     *
     * @remark Only varialbes that are required for the prediction phase will be serialized. That
     * is, saved models can be used for predicting new data points.
     *
     * @param out an output stream.
     *
     * @return the output stream.
     */
    void serialize(std::ostream& stream) const {
      ontology.serialize(stream);
      Ensemble::serialize(stream);
    }

    /**
     * @brief Loads a network ensemble model from a file.
     *
     * @param in an input stream.
     *
     * @return the input stream.
     */
    void deserialize(std::istream& stream) {
      ontology.deserialize(stream);
      Ensemble::deserialize(stream);
    }
};

#endif // leaving _LIBCB_FANNGO_H_
