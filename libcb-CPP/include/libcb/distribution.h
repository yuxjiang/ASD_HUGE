//! libcb/distribution.h

#ifndef _LIBCB_DISTRIBUTION_H_
#define _LIBCB_DISTRIBUTION_H_

#include <boost/math/distributions/hypergeometric.hpp>
#include <boost/math/distributions/students_t.hpp>
#include "util.h"

/**
 * @brief namespace of "hypothesis testing"
 */
namespace htest {

  /**
   * @brief Tail specification for hypothesis testing
   */
  enum Tail {
    BOTH, //!< both tail
    LEFT, //!< left tail
    RIGHT //!< right tail
  };

  /**
   * @brief Computes P-value for Fisher's exact test.
   *
   * @tparam T returning type, default = double.
   * @param a entry (1, 1) of the contingency table.
   * @param b entry (1, 2) of the contingency table.
   * @param c entry (2, 1) of the contingency table.
   * @param d entry (2, 2) of the contingency table.
   * @param tail which tail to integrate.
   *
   * @remark
   * Contingency table:
   * |         |   DRAWN   |       LEFT        | TOTAL |
   * | ------- | --------- | ----------------- | ----- |
   * | SUCCESS |   k (a)   |     r - k (b)     |   r   |
   * | FAILURE | n - k (c) | N + k - n - r (d) | N - r |
   * | ------- | --------- | ----------------- | ----- |
   * |  TOTAL  |     n     |       N - n       |   N   |
   * where:
   * N - population size (finite)
   * n - number of samples (n < N)
   * r - number of success in population
   * k - number of success in samples
   *
   * @return P-value.
   */
  template<typename T = double>
    T fishertest(int a, int b, int c, int d, htest::Tail tail = htest::BOTH) {
      int N = a + b + c + d;
      int r = a + b;
      int n = a + c;
      int kmin = r + n > N ? r + n - N : 0;
      int kmax = r > n ? n : r;
      boost::math::hypergeometric_distribution<T> f(r, n, N);
      T pvalue(0.0);
      if (tail == htest::LEFT) {
        for (int k = kmin; k <= a; ++k) {
          pvalue += boost::math::pdf(f, k);
        }
      } else if (tail == htest::RIGHT) {
        for (int k = a; k <= kmax; ++k) {
          pvalue += boost::math::pdf(f, k);
        }
      } else { // tail both
        T cutoff = boost::math::pdf(f, a);
        for (int k = kmin; k <= kmax; ++k) {
          T p = boost::math::pdf(f, k);
          if (p <= cutoff) {
            pvalue += p;
          }
        }
      }
      return pvalue;
    }

  /**
   * @brief (One sample) Student-t test
   *
   * @tparam T
   * @param sm mean of the sample
   * @param sd standard deviation of the sample
   * @param sn size of the sample
   * @param tail which tail to integrate:
   *
   * @return P-value.
   */
  template<typename T = double>
    T t_test(T sm, T sd, int sn, htest::Tail tail = htest::BOTH) {
      T t_stat = sm / (sd / sqrt(1.0*sn));
      T v = sn - 1;
      boost::math::students_t dist(v);
      T pvalue = 0.0;
      switch (tail) {
        case htest::BOTH: // sample mean is not zero
          pvalue = 2.0 * boost::math::cdf(boost::math::complement(dist, fabs(t_stat)));
          break;
        case htest::LEFT: // sample mean is less than zero
          pvalue = boost::math::cdf(dist, t_stat);
          break;
        case htest::RIGHT: // sample mean is greater than zero
          pvalue = boost::math::cdf(boost::math::complement(dist, t_stat));
          break;
        default:
          break;
      }
      return pvalue;
    }

  /**
   * @brief Two-sample t-test with equal variance assumption.
   *
   * @remark this snippet of code is credited to boost library.
   *
   * @tparam T
   * @param sm1 mean of sample 1
   * @param sd1 standard deviation of sample 1
   * @param sn1 size of sample 1
   * @param sm2 mean of sample 2
   * @param sd2 standard deviation of sample 2
   * @param sn2 size of sample 2
   *
   * @return P-value.
   */
  template<typename T = double>
    T two_sample_t_test_eqv(T sm1, T sd1, int sn1,
        T sm2, T sd2, int sn2,
        htest::Tail tail = htest::BOTH) {
      /* std::cerr << "[DEBUG] " << __PRETTY_FUNCTION__ << "\n" */
      /*     << "sm1: " << sm1 << "\tsd1: " << sd1 << "\tsn1: " << sn1 << "\n" */
      /*     << "sm2: " << sm2 << "\tsd2: " << sd2 << "\tsn2: " << sn2 << "\n"; */
      if (sn1 <= 1 && sn2 <= 1) {
        error_and_exit(__PRETTY_FUNCTION__, "Too few samples for both sets.");
      } else if (sn1 <= 1) {
        return t_test(sm1-sm2, sd2, sn2, tail);
      } else if (sn2 <= 1) {
        return t_test(sm1-sm2, sd1, sn1, tail);
      }
      T v = sn1 + sn2 - 2; // degrees of freedom:
      T sp = sqrt(((sn1-1) * sd1 * sd1 + (sn2-1) * sd2 * sd2) / v); // pooled variance:
      T t_stat = (sm1 - sm2) / (sp * sqrt(1.0 / sn1 + 1.0 / sn2)); // t-statistic:
      boost::math::students_t dist(v);
      T pvalue = 0.0;
      switch (tail) {
        case htest::BOTH: // mean(sample1) != mean(sample2)
          pvalue = 2.0 * boost::math::cdf(boost::math::complement(dist, fabs(t_stat)));
          break;
        case htest::LEFT: // mean(sample1) < mean(sample2)
          pvalue = boost::math::cdf(dist, t_stat);
          break;
        case htest::RIGHT: // mean(sample1) > mean(sample2)
          pvalue = boost::math::cdf(boost::math::complement(dist, t_stat));
          break;
        default:
          break;
      }
      return pvalue;
    }

  /**
   * @brief Two-sample t-test without equal variance assumption.
   *
   * @remark this snippet of code is credited to boost library.
   *
   * @tparam T
   * @param sm1 mean of sample 1
   * @param sd1 standard deviation of sample 1
   * @param sn1 size of sample 1
   * @param sm2 mean of sample 2
   * @param sd2 standard deviation of sample 2
   * @param sn2 size of sample 2
   * @param tail which tail to integrate:
   *
   * @return P-value.
   */
  template<typename T = double>
    T two_sample_t_test_noeqv(T sm1, T sd1, int sn1,
        T sm2, T sd2, int sn2,
        htest::Tail tail = htest::BOTH) {
      /* std::cerr << "[DEBUG] " << __PRETTY_FUNCTION__ << "\n" */
      /*     << "sm1: " << sm1 << "\tsd1: " << sd1 << "\tsn1: " << sn1 << "\n" */
      /*     << "sm2: " << sm2 << "\tsd2: " << sd2 << "\tsn2: " << sn2 << "\n"; */
      if (sn1 <= 1 && sn2 <= 1) {
        error_and_exit(__PRETTY_FUNCTION__, "Too few samples for both sets.");
      } else if (sn1 <= 1) {
        return t_test(sm1 - sm2, sd2, sn2, tail);
      } else if (sn2 <= 1) {
        return t_test(sm1 - sm2, sd1, sn1, tail);
      }
      T v = sd1 * sd1 / sn1 + sd2 * sd2 / sn2; // degrees of freedom:
      v *= v;
      T t1 = sd1 * sd1 / sn1;
      t1 *= t1;
      t1 /= (sn1 - 1);
      T t2 = sd2 * sd2 / sn2;
      t2 *= t2;
      t2 /= (sn2 - 1);
      v /= (t1 + t2);
      T t_stat = (sm1 - sm2) / sqrt(sd1 * sd1 / sn1 + sd2 * sd2 / sn2); // t-statistic:
      boost::math::students_t dist(v);
      T pvalue = 0.0;
      switch (tail) {
        case htest::BOTH: // mean(sample1) != mean(sample2)
          pvalue = 2.0 * boost::math::cdf(boost::math::complement(dist, fabs(t_stat)));
          break;
        case htest::LEFT: // mean(sample1) < mean(sample2)
          pvalue = boost::math::cdf(dist, t_stat);
          break;
        case htest::RIGHT: // mean(sample1) > mean(sample2)
          pvalue = boost::math::cdf(boost::math::complement(dist, t_stat));
          break;
        default:
          break;
      }
      return pvalue;
    }

  /**
   * @brief (One sample) t test.
   *
   * @remark this function simulates its MATLAB counterpart.
   *
   * @param x the data sample.
   * 
   * @return P-value.
   */
  double ttest(const arma::vec& x) {
    return t_test(arma::mean(x), arma::stddev(x), x.n_elem);
  }

  /**
   * @brief (Two sample) t test.
   *
   * @remark this function simulates its MATLAB counterpart.
   *
   * @param x sample 1.
   * @param y sample 2.
   *
   * @return P-value.
   */
  double ttest2(const arma::vec& x, const arma::vec& y) {
    return two_sample_t_test_noeqv(
        arma::mean(x),
        arma::stddev(x),
        x.n_elem,
        arma::mean(y),
        arma::stddev(y),
        y.n_elem);
  }
} // namespace htest

#endif

// -------------
// Yuxiang Jiang (yuxjiang@indiana.edu)
// Department of Computer Science
// Indiana University, Bloomington
// Last modified: Tue 25 Jun 2019 10:05:38 PM P
