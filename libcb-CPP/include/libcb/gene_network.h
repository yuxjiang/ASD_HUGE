//! libcb/gene_network.h

#ifndef _LIBCB_GENE_NETWORK_H_
#define _LIBCB_GENE_NETWORK_H_

#include "util.h"
#include "graph.h"
#include "sequence.h"

using GeneNetwork = SGraph<Gene>;

/**
 * @brief Creates a gene network without edges
 *
 * @param nFile a file of gene Entrez IDs.
 *
 * @return A gene network of type SGraph<Gene>.
 */
GeneNetwork load_gene_network_by_nodes(const std::string& nFile);

/**
 * @brief Loads a gene network from an plain-text edge file.
 *
 * @param eFile a plain-text edge file which has the format:
 * `<Entrez source> <Entrez destination> <weight>`
 *
 * @return A gene network of type SGraph<Gene>.
 *
 * @sa save_gene_network()
 */
GeneNetwork load_gene_network_by_edges(const std::string& eFile, char delim = '\t');

/**
 * @brief Loads a gene network from both (plain-text) node file and edges file.
 *
 * @param nFile a plain-text node file which has one EntrezID per line.
 * @param eFile a plain-text edge file which has the format:
 * `<Entrez source> <Entrez destination> <weight>`
 *
 * @return A gene network of type SGraph<Gene>.
 */
GeneNetwork load_gene_network_by_nodes_and_edges(
    const std::string& nFile,
    const std::string& eFile,
    char delim = '\t');

/**
 * @brief Saves edges to a plain-text file.
 *
 * @param eFile an output file name.
 * @param gn a gene network of type SGraph<Gene>.
 *
 * @sa load_gene_network()
 */
void save_gene_network(const std::string& eFile, const GeneNetwork& gn);

/**
 * @brief Prints a small gene network (for debugging purpose).
 *
 * @param gn a gene network of type SGraph<Gene>.
 */
void print_small_gene_network(const GeneNetwork& gn);

/**
 * Network-based propagation methods
 */
namespace nbp {
  using score_type = float;

  const score_type inf_score = std::numeric_limits<score_type>::infinity();
  const score_type eps = 1e-8;

  enum algo_type {
    DUMMY,              //!< Dummy (identity) function, not implemented.
    FUNCTIONAL_FLOW,    //!< Functional-flow
    MARKOV_RANDOM_FIELD //!< Markov random field
  };

  /**
   * @brief Network-based predictor functor
   */
  template<typename T>
    class Algorithm {
      public:
        virtual std::unordered_map<T, score_type> operator()(
            const SGraph<T>&,
            const std::vector<T>&) const = 0;

        virtual ~Algorithm() {}
    };

  /**
   * @brief Functional-flow algorithm
   */
  template<typename T>
    class FunctionalFlow : public Algorithm<T> {
      const index_t DEFAULT_NUM_ITER = 5;
      public:
      /**
       * @brief Flow capacity type.
       */
      enum capacity_type {
        CAP_RAW, //!< use raw edge weight as the flow capacity.
        CAP_NI,  //!< use the normalized in-weight as the flow capacity.
        CAP_NO   //!< use the normalized out-weight as the flow capacity.
      };

      /**
       * @brief Default constructor. (with default parameters)
       */
      FunctionalFlow() : m_ctype(CAP_RAW), m_d(DEFAULT_NUM_ITER) {}

      /**
       * @brief Constructor with parameters.
       *
       * @param ctype the capacity type.
       * @param d the number of iterations.
       */
      FunctionalFlow(capacity_type ctype, index_t d) : m_ctype(ctype), m_d(d) {}

      /**
       * @brief Implementation of apply operator()
       *
       * @param g the gene network.
       * @param positives the vector of positive annotations.
       *
       * @return Prediction scores a map from `T` to `score_type`.
       */
      std::unordered_map<T, score_type> operator()(
          const SGraph<T>& g,
          const std::vector<T>& positives) const;

      /**
       * @brief Set capacity type of functional-flow algorithm.
       *
       * @param ctype the capacity type.
       */
      void set_capacity_type(capacity_type ctype) { m_ctype = ctype; }

      /**
       * @brief Set the number of iterations of functional-flow algorithm.
       *
       * @param d the number of iterations.
       */
      void set_d(index_t d) { m_d = d; }

      private:
      /**
       * @brief Capacity type.
       */
      capacity_type m_ctype;

      /**
       * @brief Number of iterations.
       */
      index_t m_d;
    };

  template<typename T>
    class MarkovRandomField : public Algorithm<T> {
      const index_t DEFAULT_BURNIN     = 100;
      const index_t DEFAULT_LAG_PERIOD = 10;
      const index_t DEFAULT_NUM_SIM    = 2000;
      public:
      union param_t{
        float pi; // class prior
        float alpha;
        float beta;
        float gamma;
        index_t burnIn;
        index_t lagPeriod;
        index_t numSim;
      };

      MarkovRandomField() {
        m_param.pi = 0.0;
        m_param.alpha = m_param.beta = m_param.gamma = 0.0;
        m_param.burnIn = m_param.lagPeriod = m_param.numSim = 0;
      }

      MarkovRandomField(float pi, float alpha, float beta, float gamma) {
        m_param.pi    = pi;
        m_param.alpha = alpha;
        m_param.beta  = beta;
        m_param.gamma = gamma;
        // default parameters for Gibbs sampler
        m_param.burnIn    = DEFAULT_BURNIN;
        m_param.lagPeriod = DEFAULT_LAG_PERIOD;
        m_param.numSim    = DEFAULT_NUM_SIM;
      }

      MarkovRandomField(
          float pi,
          float alpha,
          float beta,
          float gamma,
          index_t bi,
          index_t lp,
          index_t ns) {
        m_param.pi        = pi;
        m_param.alpha     = alpha;
        m_param.beta      = beta;
        m_param.gamma     = gamma;
        m_param.burnIn    = bi;
        m_param.lagPeriod = lp;
        m_param.numSim    = ns;
      }

      std::unordered_map<T, score_type> operator()(
          const SGraph<T>& g,
          const std::vector<T>& positives) const;

      void set_gibbs_parameters(index_t burnIn, index_t lagPeriod, index_t numSim) {
        m_param.burnIn    = burnIn;
        m_param.lagPeriod = lagPeriod;
        m_param.numSim    = numSim;
      }

      void set_theta(float alpha, float beta, float gamma) {
        m_param.alpha = alpha;
        m_param.beta  = beta;
        m_param.gamma = gamma;
      }

      void set_pi(float pi) {
        m_param.pi = pi;
      }

      private:
      param_t m_param;
    };

  template<typename T>
    std::unordered_map<T, score_type> FunctionalFlow<T>::operator()(
        const SGraph<T>& g,
        const std::vector<T>& positives) const {
      using namespace std;
      unordered_map<T, score_type> prediction;
      vector<T> sequences = g.get_vertices();
      index_t n = g.number_of_vertices();
      unordered_map<T, index_t> s2i;
      for (index_t i = 0; i < n; ++i) {
        s2i[sequences[i]] = i;
      }
      vector<score_type> R(n, 0.0); // positive reserviors
      // Cumulative positive flows
      // Note that cummulated reserviors could never be infinity.
      vector<score_type> f(n, 0.0);
      // mark known genes
      index_t valid_count(0); // number of positive genes in network.
      for (const auto& seq : positives) {
        if (CONTAINS(s2i, seq)) {
          R[s2i.at(seq)] = inf_score;
          ++valid_count;
        }
      }
      if (valid_count == 0) {
        warn(__PRETTY_FUNCTION__, "No positives in the networks.");
      } else {
        if (valid_count < 10) {
          warn(__PRETTY_FUNCTION__, "Less then 10 positives in the network.");
        }
        // pre-compute normalized out-going weight matrix
        vector<vector<float>> wm_normed_o = g.out_degree_normalized_adjacency_matrix();
        // if capacity type is set of normalized-out
        vector<vector<float>> wm_normed_i;
        if (m_ctype == FunctionalFlow::CAP_NI) {
          wm_normed_i = g.in_degree_normalized_adjacency_matrix();
        }
        // functional flow
        vector<typename SGraph<T>::weighted_edge_type> weighted_edges = g.get_weighted_edges();
        for (index_t d = 0; d < m_d; ++d) {
          vector<score_type> delta_R(n, 0.0);
          index_t isrc, idst;
          typename SGraph<T>::weight_type weight;
          for (const auto& weighted_edge : weighted_edges) {
            std::tie(isrc, idst, weight) = weighted_edge;
            // determine pipe capacity
            score_type cap_i, cap_o;
            if (m_ctype == FunctionalFlow::CAP_RAW) {
              cap_i = cap_o = weight;
            } else if (m_ctype == FunctionalFlow::CAP_NO) {
              cap_i = wm_normed_o[idst][isrc];
              cap_o = wm_normed_o[isrc][idst];
            } else if (m_ctype == FunctionalFlow::CAP_NI) {
              cap_i = wm_normed_i[idst][isrc];
              cap_o = wm_normed_i[isrc][idst];
            } else {
              error_and_exit(__PRETTY_FUNCTION__, "Unknown capacity type.");
            }
            // run a flow
            if (R[isrc] == R[idst] || std::fabs(R[isrc] - R[idst]) < nbp::eps) {
              // no flow, (note: c++11 treats 'inf == inf' to be true)
              continue;
            } else if (R[isrc] > R[idst]) {
              // out-going flow: src -> dst
              score_type flow = std::min(cap_o, R[isrc] * wm_normed_o[isrc][idst]);
              delta_R[isrc] -= flow;
              delta_R[idst] += flow;
              f[idst] += flow; // f only accumulate incoming flows
            } else {
              // in-coming flow: src <- dst
              score_type flow = std::min(cap_i, R[idst] * wm_normed_o[idst][isrc]);
              delta_R[isrc] += flow;
              delta_R[idst] -= flow;
              f[isrc] += flow;
            }
          }
          // update R with changes
          for (index_t i = 0; i < n; ++i) {
            R[i] += delta_R[i];
          }
        }
      }
      // map results
      for (index_t i = 0; i < n; ++i) {
        prediction[sequences[i]] = f[i];
      }
      return prediction;
    }

  template<typename T>
    std::unordered_map<T, score_type> MarkovRandomField<T>::operator()(
        const SGraph<T>& g,
        const std::vector<T>& positives) const {
      using namespace std;
      unordered_map<T, score_type> prediction;
      vector<T> sequences = g.get_vertices();
      index_t n = g.number_of_vertices();
      unordered_map<T, index_t> s2i;
      for (index_t i = 0; i < n; ++i) {
        s2i[sequences[i]] = i;
      }
      vector<index_t> posIndex, unkIndex;
      unordered_set<T> posSet(positives.begin(), positives.end());
      for (index_t i = 0; i < n; ++i) {
        if (!CONTAINS(posSet, sequences[i])) {
          unkIndex.push_back(i);
        }
      }
      std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
      std::uniform_real_distribution<double> distribution(0.0, 1.0);
      // set the status for positives to 1
      vector<index_t> status(n, 0), nextStatus(n, 0);
      for (auto pos : positives) {
        status[s2i[pos]] = 1;
        prediction[pos]  = 1.0; // fix prediction to be 1
      }
      // randomly initialize the status for unknowns based on class prior
      for (auto i : unkIndex) {
        if (distribution(generator) < m_param.pi) {
          status[i] = 1;
        }
      }
      // find neighbors once ahead of time
      vector<vector<index_t>> neighbors;
      for (index_t i = 0; i < n; ++i) {
        neighbors[i] = g.neighbor_of(i);
      }
      vector<score_type> prob(n, 0.0);
      index_t numSamples = 0;
      for (index_t t = 0; t < m_param.burnIn + m_param.numSim; ++t) {
        // compute next status
        for (auto i : unkIndex) {
          float M0(0.0), M1(0.0);
          for (auto j : neighbors[i]) {
            if (status[j] == 0) {
              M0 += 1.0;
            } else {
              M1 += 1.0;
            }
          }
          float e = exp(m_param.alpha + (m_param.beta - 1.0) * M0 + (m_param.gamma - m_param.beta) * M1);
          if (distribution(generator) < (e / (1 + e))) {
            nextStatus[i] = 1;
          } else {
            nextStatus[i] = 0;
          }
        }
        // update status
        for (auto i : unkIndex) {
          status[i] = nextStatus[i];
        }
        if ((t > m_param.burnIn) && (t % m_param.lagPeriod != 0)) {
          // collect sample and update probability
          for (auto k : unkIndex) {
            prob[k] = static_cast<float>(numSamples * prob[k] + status[k]) / (numSamples + 1.0);
          }
          numSamples += 1;
        }
      }
      for (auto i : unkIndex) {
        prediction[sequences[i]] = prob[i];
      }
      return prediction;
    }
}

#endif // _LIBCB_GENE_NETWORK_H_

// -------------
// Yuxiang Jiang (yuxjiang@indiana.edu)
// Department of Computer Science
// Indiana University, Bloomington
// Last modified: Tue 25 Jun 2019 11:27:55 PM P
