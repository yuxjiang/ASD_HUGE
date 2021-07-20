//! libcb/evaluation.h

#ifndef _LIBCB_EVALUATION_H_
#define _LIBCB_EVALUATION_H_

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <random>
#include <chrono>
#include "util.h"

/**
 * @brief a class of 2D points.
 */
struct Point {
  //! @a x coordinate.
  double x;

  //! @a y coordinate.
  double y;

  //! The default constructor.
  Point() : x(0.0), y(0.0) {}

  //! A constructor given another Point.
  Point(const Point& pt) : x(pt.x), y(pt.y) {}

  //! A constructor given @a x and @a y.
  Point(double _x, double _y) : x(_x), y(_y) {}

  const Point& operator=(const Point& pt) {
    if (this != &pt) {
      x = pt.x;
      y = pt.y;
    }
    return *this;
  }

  Point operator+(const Point& pt) const {
    Point res = (*this);
    res.x += pt.x;
    res.y += pt.y;
    return res;
  }

  Point operator-(const Point& pt) const {
    Point res = (*this);
    res.x -= pt.x;
    res.y -= pt.y;
    return res;
  }

  const Point& operator+=(const Point& pt) {
    x += pt.x;
    y += pt.y;
    return *this;
  }

  const Point& operator-=(const Point& pt) {
    x -= pt.x;
    y -= pt.y;
    return *this;
  }

  Point operator*(double s) const {
    Point res = (*this);
    res.x *= s;
    res.y *= s;
    return res;
  }

  Point operator/(double s) const {
    if (s == 0) {
      error_and_exit(__PRETTY_FUNCTION__, "Division by zero.");
    }
    Point res = (*this);
    res.x /= s;
    res.y /= s;
    return res;
  }

  const Point& operator*=(double s) {
    x *= s;
    y *= s;
    return *this;
  }

  const Point& operator/=(double s) {
    if (s == 0) {
      error_and_exit(__PRETTY_FUNCTION__, "Division by zero.");
    }
    x /= s;
    y /= s;
    return *this;
  }

  bool operator==(const Point& pt) const {
    return x == pt.x && y == pt.y;
  }
};

/**
 * @brief a class of confusion matrix
 */
struct ConfusionMatrix {
  // note that we use "double" instead of "index_t" to accomodate the weighted
  // scenarios, i.e., weighted fmax, semantic distance, etc.
  double TP, FP, TN, FN;
  ConfusionMatrix() {
    TP = FP = TN = FN = 0.0;
  }

  ConfusionMatrix(double tp, double fp, double tn, double fn) {
    TP = tp;
    FP = fp;
    TN = tn;
    FN = fn;
  }

  double precision() const {
    return TP / (TP + FP);
  }

  double recall() const {
    return TP / (TP + FN);
  }

  double sensitivity() const {
    return recall();
  }

  double specificity() const {
    return TN / (TN + FP);
  }

  double tpr() const {
    return recall();
  }

  double fpr() const {
    return FP / (TN + FP);
  }

  // remaining uncertainty
  double ru() const {
    return FN;
  }

  // misinformation
  double mi() const {
    return FP;
  }

  // normalized remaining uncertainty
  double nru() const {
    return FN / (TP + FN + FP);
  }

  // normalized misinformation
  double nmi() const {
    return FP / (TP + FP + FN);
  }

  const ConfusionMatrix& operator=(const ConfusionMatrix& other) {
    if (this != &other) {
      TP = other.TP;
      FP = other.FP;
      TN = other.TN;
      FN = other.FN;
    }
    return *this;
  }

  ConfusionMatrix operator+(const ConfusionMatrix& other) const {
    ConfusionMatrix res = *this;
    res += other;
    return res;
  }

  const ConfusionMatrix& operator+=(const ConfusionMatrix& other) {
    TP += other.TP;
    FP += other.FP;
    TN += other.TN;
    FN += other.FN;
    return *this;
  }

  void print() const {
    std::cout << "--------" << std::endl;
    std::cout << std::setprecision(4) << "TP: " << std::setw(8) << TP << "\tFP: " << FP << std::endl;
    std::cout << std::setprecision(4) << "FN: " << std::setw(8) << FN << "\tTN: " << TN << std::endl;
  }
};

/**
 * @brief a class of cross-validation folds.
 */
class CVFolds {
  private:
    //! The number of data points.
    index_t m_n;

    //! The number of folds.
    index_t m_v;

    /**
     * @brief Fold index.
     *
     * `m_fid[i] = j` means the `i`-th data points belongs to test fold `j`.
     */
    std::vector<index_t> m_fid;

  public:
    //! The default constructor.
    CVFolds() : m_n(0), m_v(0) {}

    /**
     * @brief A constructor with `n` and `v`.
     *
     * @param n the number of data points.
     * @param v the number of folds.
     * @param seed a random seed, generated from current time if set to 0.
     */
    CVFolds(index_t n, index_t v, unsigned seed = 0) : m_n(n), m_v(v) {
      using namespace std::chrono;
      m_fid = std::vector<index_t>(n, 0);
      for (index_t i = 0; i < n; ++i) {
        m_fid[i] = (i % v);
      }
      if (seed == 0) {
        seed = system_clock::now().time_since_epoch().count();
      }
      std::shuffle(m_fid.begin(), m_fid.end(), std::default_random_engine(seed));
    }

    /**
     * @brief Updates `n` and `v`.
     *
     * @param n the number of data points.
     * @param v the number of folds.
     * @param seed a random seed, generated from current time if set to 0.
     */
    void update(index_t n, index_t v, unsigned seed = 0) {
      using namespace std::chrono;
      m_fid.resize(n);
      for (index_t i = 0; i < n; ++i) {
        m_fid[i] = i % v;
      }
      if (seed == 0) {
        seed = system_clock::now().time_since_epoch().count();
      }
      std::shuffle(m_fid.begin(), m_fid.end(), std::default_random_engine(seed));
    }

    /**
     * @brief Gets one training fold indices.
     *
     * @param index the `i`-th fold.
     *
     * @return The vector of indices.
     */
    std::vector<index_t> training_fold(index_t index) const {
      std::vector<index_t> fold;
      for (index_t i = 0; i < m_n; ++i) {
        if (m_fid[i] != index) {
          fold.push_back(i);
        }
      }
      return fold;
    }

    /**
     * @brief Gets one test fold indices.
     *
     * @param index the `i`-th fold.
     *
     * @return The vector of indices.
     */
    std::vector<index_t> test_fold(index_t index) const {
      std::vector<index_t> fold;
      for (index_t i = 0; i < m_n; ++i) {
        if (m_fid[i] == index) {
          fold.push_back(i);
        }
      }
      return fold;
    }

    /**
     * @brief Gets the number of cross-validation folds.
     *
     * @return The number of folds.
     */
    index_t number_of_folds() const { return m_n; }

    /**
     * @brief Gets the number of data points.
     *
     * @return The number of data points.
     */
    index_t number_of_data_points() const { return m_n; }

    /**
     * @brief Writes to a file.
     *
     * @param stream An output stream.
     */
    void serialize(std::ostream stream) const {
      uint32_t data;
      data = m_n;
      SAVE_VAR(stream, data);
      data = m_v;
      SAVE_VAR(stream, data);
      for (const auto& index: m_fid) {
        data = index;
        SAVE_VAR(stream, data);
      }
    }

    /**
     * @brief Reads from a file.
     *
     * @param stream An input stream.
     */
    void deserialize(std::istream stream) {
      uint32_t data;
      LOAD_N(stream, data, sizeof(uint32_t));
      m_n = data;
      LOAD_N(stream, data, sizeof(uint32_t));
      m_v = data;
      m_fid.resize(m_n);
      for (index_t i = 0; i < m_n; ++i) {
        LOAD_N(stream, data, sizeof(uint32_t));
        m_fid[i] = data;
      }
    }
};

/**
 * @brief Computes @a trapz of a collection of points.
 *
 * @remark Unlike MATLAB version of @a trapz, it sorts points according to its
 * @a x part in ascending order and then compute the trapz.
 *
 * @param pts a vector of points.
 *
 * @return The trapezoidal numerical integration.
 */
double ordered_trapz(const std::vector<Point>& pts);

/**
 * @brief Computes the maximum F_{beta}-measure of a collection of points.
 *
 * @param pts a vector of points.
 * @param beta the F-measure parameter, default: 1 (F1-measure/F1-score).
 *
 * @return The maximum F-measure.
 */
double fmax(const std::vector<Point>& pts, double beta = 1.0);

/**
 * @brief Computes the minimum semantic distance of a collection of points.
 *
 * @param pts a vector of points.
 *
 * @return The minimum SD.
 */
double sdmin(const std::vector<Point>& pts);

template<typename T = double>
std::vector<ConfusionMatrix> get_cms_with_threshold(const std::vector<T>& p, const std::vector<bool>& t, const std::set<T> taus, const std::vector<double>& weights) {
  if (p.size() != t.size()) {
    error_and_exit(__PRETTY_FUNCTION__, "Inconsistent number of predictions and labels.");
  }
  index_t n = p.size();
  double P = 0; // total weights of positives
  double N = 0; // total weights of negatives
  for (index_t i = 0; i < n; ++i) {
    if (t[i]) P += weights[i];
    else N += weights[i];
  }
  arma::uvec order = arma::sort_index(arma::Col<T>(p), "ascend");
  std::vector<ConfusionMatrix> cms;
  double TP = P, FP = N; // weighted TP and FP
  index_t index = 0; // an index into data points
  // going from small to large predicted values
  for (const auto& tau : taus) {
    while (index < n && p[order(index)] < tau) {
      if (t[order(index)]) {
        TP -= weights[order(index)];
      } else {
        FP -= weights[order(index)];
      }
      index ++;
    }
    cms.emplace_back(TP, FP, N - FP, P - TP);
  }
  return cms;
}

template<typename T = double>
std::vector<ConfusionMatrix> get_cms_with_threshold(const std::vector<T>& p, const std::vector<bool>& t, const std::set<T> taus) {
  return get_cms_with_threshold(p, t, taus, std::vector<double>(p.size(), 1.0));
}

/**
 * @brief Computes the receiver operating characteristic curve with given
 * thresholds.
 *
 * @pre The prediction type T must have a properly defined "less-than" operator.
 * E.g., float, double.
 *
 * @remark The number of (fpr, tpr) points will be the same as the number of
 * thresholds, tau.
 *
 * @tparam T
 * @param p a vector of predictions with type T.
 * @param t a vector of binary ground-truth of type boolean.
 * @param taus a set of thresholds of type T.
 *
 * @return The ROC curve.
 */
template<typename T = double>
std::vector<Point> get_roc_with_thresholds(const std::vector<T>& p, const std::vector<bool>& t, const std::set<T> taus) {
  auto cms = get_cms_with_threshold<T>(p, t, taus);
  std::vector<Point> curve;
  for (const auto& cm : cms) {
    curve.emplace_back(cm.fpr(), cm.tpr());
  }
  return curve;
}

/**
 * @brief Computes the receiver operating characteristic curve.
 *
 * @remark
 * @arg The prediction type T must have a properly defined "less-than" operator.
 * E.g., float, double.
 * @arg Threasholds are set as all unique predicted scores.
 *
 * @tparam T
 * @param p a vector of predictions with type T.
 * @param t a vector of binary ground-truth of type boolean.
 * @param taus a set of thresholds of type T.
 *
 * @return The ROC curve.
 */
template<typename T = double>
std::vector<Point> get_roc(const std::vector<T>& p, const std::vector<bool>& t) {
  std::set<T> taus(p.begin(), p.end());
  return get_roc_with_thresholds(p, t, taus);
}

/**
 * @brief Computes Area under ROC (AUC) of a prediction with given thresholds.
 *
 * @pre The prediction type T must have a properly defined "less-than" operator.
 * E.g., float, double.
 *
 * @tparam T
 * @param p a vector of predictions with type T.
 * @param t a vector of binary ground-truth of type boolean.
 * @param taus a set of thresholds of type T.
 *
 * @return The AUC.
 */
template<typename T = double>
double get_auc_with_thresholds(const std::vector<T>& p, const std::vector<bool>& t, const std::set<T>& taus) {
  return ordered_trapz(get_roc_with_thresholds(p, t, taus)); // AUC
}

/**
 * @brief Computes Area under ROC (AUC) of a prediction.
 *
 * @remark
 * @arg The prediction type T must have a properly defined "less-than" operator.
 * E.g., float, double.
 * @arg Threasholds are set as all unique predicted scores.
 *
 * @tparam T
 * @param p a vector of predictions with type T.
 * @param t a vector of binary ground-truth of type boolean.
 *
 * @return The AUC.
 *
 * @sa get_auc_with_thresholds()
 */
template<typename T = double>
double get_auc(const std::vector<T>& p, const std::vector<bool>& t) {
  std::set<T> taus(p.begin(), p.end());
  return get_auc_with_thresholds(p, t, taus);
}

/**
 * @brief Computes the precision-recall curve with given thresholds.
 *
 * @pre The prediction type T must have a properly defined "less-than" operator.
 * E.g., float, double.
 *
 * @remark The number of (recall, precision) points will be the same as the
 * number of thresholds, tau.
 *
 * @tparam T
 * @param p a vector of predictions with type T.
 * @param t a vector of binary ground-truth of type boolean.
 * @param taus a set of thresholds of type T.
 *
 * @return The precision-recall curve.
 */
template<typename T = double>
std::vector<Point> get_prc_with_thresholds(const std::vector<T>& p, const std::vector<bool>& t, const std::set<T> taus) {
  auto cms = get_cms_with_threshold<T>(p, t, taus);
  std::vector<Point> curve;
  for (const auto& cm : cms) {
    curve.emplace_back(cm.fpr(), cm.tpr());
  }
  std::reverse(curve.begin(), curve.end()); // reverse points to have "recall" sorted in ascending order
  return curve;
}

/**
 * @brief Computes the precision-recall curve.
 *
 * @remark
 * @arg The prediction type T must have a properly defined "less-than" operator.
 * E.g., float, double.
 * @arg Threasholds are set as all unique predicted scores.
 *
 * @tparam T
 * @param p a vector of predictions with type T.
 * @param t a vector of binary ground-truth of type boolean.
 *
 * @return The precision-recall curve.
 */
template<typename T = double>
std::vector<Point> get_prc(const std::vector<T>& p, const std::vector<bool>& t) {
  std::set<T> taus(p.begin(), p.end());
  return get_prc_with_thresholds(p, t, taus);
}

/**
 * @brief Computes the ru-mi curve (remaining uncertainty, misinformation) with
 * given thresholds
 *
 * @remark
 * @arg The prediction type T must have a properly defined "less-than" operator.
 * E.g., float, double.
 * @arg Threasholds are set as all unique predicted scores.
 *
 * @remark the resulting (ru, mi) will be normalized automatically.
 *
 * @remark The number of (ru, mi) points will be the same as the
 * number of thresholds, tau.
 *
 * @tparam T
 * @param p a vector of predictions with type T.
 * @param t a vector of binary ground-truth of type boolean.
 * @param ia a vector of information accretion for each item (term) should have
 * the same length as p and t.
 *
 * @return The precision-recall curve.
 */
template<typename T = double>
std::vector<Point> get_rmc_with_thresholds(const std::vector<T>& p, const std::vector<bool>& t, const std::vector<double>& ia, const std::set<T> taus) {
  auto cms = get_cms_with_threshold<T>(p, t, taus);
  std::vector<Point> curve;
  for (const auto& cm : cms) {
    curve.emplace_back(cm.nru(), cm.nmi());
  }
  return curve;
}

/**
 * @brief Computes the ru-mi curve (remaining uncertainty, misinformation)
 *
 * @remark
 * @arg The prediction type T must have a properly defined "less-than" operator.
 * E.g., float, double.
 * @arg Threasholds are set as all unique predicted scores.
 *
 * @tparam T
 * @param p a vector of predictions with type T.
 * @param t a vector of binary ground-truth of type boolean.
 * @param ia a vector of information accretion for each item (term) should have
 * the same length as p and t.
 *
 * @return The precision-recall curve.
 */
template<typename T = double>
std::vector<Point> get_rmc(const std::vector<T>& p, const std::vector<bool>& t, const std::vector<double>& ia) {
  std::set<T> taus(p.begin(), p.end());
  return get_rmc_with_thresholds(p, t, ia, taus);
}

/**
 * @brief Computes the maximum F1-measure with given thresholds.
 *
 * @tparam T
 * @param p a vector of predictions with type T.
 * @param t a vector of binary ground-truth of type boolean.
 * @param taus a set of thresholds of type T.
 *
 * @return the maximum F1-measure.
 */
template<typename T = double>
double get_fmax_with_thresholds(const std::vector<T>& p, const std::vector<bool>& t, const std::set<T> taus) {
  return fmax(get_prc_with_thresholds(p, t, taus));
}

/**
 * @brief Computes the maximum F1-measure.
 *
 * @tparam T
 * @param p a vector of predictions with type T.
 * @param t a vector of binary ground-truth of type boolean.
 *
 * @return the maximum F1-measure.
 */
template<typename T = double>
double get_fmax(const std::vector<T>& p, const std::vector<bool>& t) {
  return fmax(get_prc(p, t));
}

template<typename T = double>
double get_sdmin(const std::vector<T>& p, const std::vector<bool>& t, const std::vector<double>& ia) {
  return sdmin(get_rmc(p, t, ia));
}

enum EvalCentric {
  EC_INS, // instance (protein) centric
  EC_LAB  // label (term) centric
};

enum EvalMetric {
  EM_FMAX,
  EM_PRC,
  EM_SDMIN,
  EM_RMC,
  EM_AUC,
  EM_ROC
};

/**
 * @brief Computes the macro-avarage of a given metric.
 *
 * @param P the predicted matrix.
 * @param T the ground-truth matrix.
 * @param centric instance-/label-centric.
 * @param metric the metric of interest.
 * @param weights the weights for each object.
 *
 * @return the macro-averaged metric.
 */
double macro_average(const arma::mat& P, const arma::umat& T, EvalCentric centric, EvalMetric metric, const std::vector<double>& weights);

double macro_average(const arma::mat& P, const arma::umat& T, EvalCentric centric, EvalMetric metric);

/**
 * @brief Computes the micro-average of a given curve metric.
 *
 * @param P the predicted matrix.
 * @param T the ground-truth matrix.
 * @param centric instance-/label-centric.
 * @param metric the metric of interest.
 * @param weights the weights for each object.
 *
 * @return the micro-averaged (curve) metric.
 */
std::vector<Point> micro_average_curve(const arma::mat& P, const arma::umat& T, EvalCentric centric, EvalMetric metric, const std::vector<double>& weights);

std::vector<Point> micro_average_curve(const arma::mat& P, const arma::umat& T, EvalCentric centric, EvalMetric metric);

/**
 * @brief Computes the micro-avarage of a given metric.
 *
 * @param P the predicted matrix.
 * @param T the ground-truth matrix.
 * @param centric instance-/label-centric.
 * @param metric the metric of interest.
 * @param weights the weights for each object.
 *
 * @return the micro-averaged metric.
 */
double micro_average(const arma::mat& P, const arma::umat& T, EvalCentric centric, EvalMetric metric, const std::vector<double>& weights);

double micro_average(const arma::mat& P, const arma::umat& T, EvalCentric centric, EvalMetric metric);

#endif // _LIBCB_EVALUATION_H_

// end of file

// -------------
// Yuxiang Jiang (yuxjiang@indiana.edu)
// Department of Computer Science
// Indiana University, Bloomington
// Last modified: Tue 25 Jun 2019 11:27:46 PM P
