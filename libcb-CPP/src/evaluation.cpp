//! libcb/evaluation.cpp

#include <algorithm>
#include <limits>
#include <cmath>

#include <libcb/util.h>
#include <libcb/evaluation.h>

using namespace std;
using namespace arma;

typedef std::vector<double> stdvec;
typedef std::vector<bool>   stdbvec;

// for local usage: convert arma::urowvec to std::vector<bool>
vector<bool> convert_to_bvec(const urowvec& row) {
    vector<bool> res(row.n_elem, false);
    for (index_t i = 0; i < row.n_elem; ++i) {
        if (row(i) > 0) res[i] = true;
    }
    return res;
}

double ordered_trapz(const vector<Point>& pts) {
    if (pts.size() < 2) return 0.0;
    vector<Point> sorted_pts = pts;
    // sort points according to x, and then y
    sort(sorted_pts.begin(), sorted_pts.end(), [](const Point& a, const Point& b) {
            return (a.x == b.x) ? (a.y < b.y) : (a.x < b.x);
            });
    // remove consecutive repetitives
    auto it = unique(sorted_pts.begin(), sorted_pts.end());
    sorted_pts.resize(distance(sorted_pts.begin(), it));
    double area(0.0);
    for (index_t i = 1; i < sorted_pts.size(); ++i) {
        area += (sorted_pts[i].y + sorted_pts[i - 1].y) * (sorted_pts[i].x - sorted_pts[i - 1].x);
    }
    return 0.5 * area;
}

double fmax(const vector<Point>& pts, double beta) {
    if (beta <= 0) {
        error_and_exit(__PRETTY_FUNCTION__, "Beta must be positive.");
    }
    double mx = 0.0;
    double beta2 = beta * beta;
    for (const auto& pt : pts) {
        double score = (pt.x * pt.y) / (beta2 * pt.y + pt.x);
        mx = max(mx, score);
    }
    return (1.0 + beta2) * mx;
}

double sdmin(const vector<Point>& pts) {
    double mn = numeric_limits<double>::max();
    for (const auto& pt : pts) {
        double sd2 = pt.x * pt.x + pt.y * pt.y;
        mn = min(mn, sd2);
    }
    return sqrt(mn);
}

double macro_average(const mat& P, const umat& T, EvalCentric centric, EvalMetric metric, const vector<double>& weights) {
    mat p;  // local copy of P and T
    umat t;
    if (centric == EC_LAB) {
        if (metric == EM_SDMIN || metric == EM_RMC) {
            error_and_exit(__PRETTY_FUNCTION__, "Cannot choose this metric for label-centric view.");
        }
        p = P.t();
        t = T.t();
    } else {
        p = P;
        t = T;
    }
    if (p.n_rows != t.n_rows || p.n_cols != t.n_cols || p.n_cols != weights.size()) {
        error_and_exit(__PRETTY_FUNCTION__, "dimension mismatch.");
    }
    double avg = 0.0;
    index_t counts = p.n_rows;
    switch (metric) {
        case EM_FMAX:
            for (index_t i = 0; i < counts; ++i) {
                avg += get_fmax(conv_to< stdvec >::from(p.row(i)), convert_to_bvec(t.row(i)));
            }
            break;
        case EM_SDMIN:
            for (index_t i = 0; i < counts; ++i) {
                avg += get_sdmin(conv_to< stdvec >::from(p.row(i)), convert_to_bvec(t.row(i)), weights);
            }
            break;
        case EM_AUC:
            for (index_t i = 0; i < counts; ++i) {
                avg += get_auc(conv_to< stdvec >::from(p.row(i)), convert_to_bvec(t.row(i)));
            }
            break;
        default:
            break;
    }
    return avg / static_cast<double>(counts);
}

double macro_average(const mat& P, const umat& T, EvalCentric centric, EvalMetric metric) {
    vector<double> weights;
    if (centric == EC_INS) {
        weights = vector<double>(P.n_cols, 1.0);
    } else {
        weights = vector<double>(P.n_rows, 1.0);
    }
    return macro_average(P, T, centric, metric, weights);
}

vector<Point> micro_average_curve(const mat& P, const umat& T, EvalCentric centric, EvalMetric metric, const vector<double>& weights) {
    mat p;  // local copy of P and T
    umat t;
    if (centric == EC_LAB) {
        if (metric == EM_SDMIN || metric == EM_RMC) {
            error_and_exit(__PRETTY_FUNCTION__, "Cannot choose this metric for label-centric view.");
        }
        p = P.t();
        t = T.t();
    } else {
        p = P;
        t = T;
    }
    if (p.n_rows != t.n_rows || p.n_cols != t.n_cols || p.n_cols != weights.size()) {
        error_and_exit(__PRETTY_FUNCTION__, "dimension mismatch.");
    }
    index_t num_q = 101; // number of quantiles (percentiles)
    vec qs = arma::linspace<vec>(0.0, 1.0, num_q);

    // use quantile (percentile) instead of a fixed array: [0.01:0.01:1.00] as cutoffs
    set<double> taus;
    vector<double> all_scores = arma::conv_to< vector<double> >::from(P.as_col());
    // sort all scores and remove 0.00
    sort(all_scores.begin(), all_scores.end());
    all_scores.erase(all_scores.begin(), upper_bound(all_scores.begin(), all_scores.end(), 0));
    for (const auto& q: qs) {
        taus.insert(quantile(all_scores, q, true));
    }
    taus.insert(0.0);
    taus.insert(1.0);
    index_t num_taus = taus.size();
    index_t n = p.n_rows; // number of "curves"
    vector<ConfusionMatrix> accu_cms(num_taus);
    for (index_t i = 0; i < n; ++i) {
        auto cms = get_cms_with_threshold(conv_to< stdvec >::from(p.row(i)), convert_to_bvec(t.row(i)), taus, weights);
        for (index_t k = 0; k < num_taus; ++k) {
            accu_cms[k] += cms[k];
        }
    }
    vector<Point> avg_curve;
    switch (metric) {
        case EM_PRC:
            for (index_t i = 0; i < num_taus; ++i) {
                avg_curve.emplace_back(accu_cms[i].recall(), accu_cms[i].precision());
            }
            std::reverse(avg_curve.begin(), avg_curve.end());
            break;
        case EM_RMC:
            for (index_t i = 0; i < num_taus; ++i) {
                avg_curve.emplace_back(accu_cms[i].nru(), accu_cms[i].nmi());
            }
            break;
        case EM_ROC:
            for (index_t i = 0; i < num_taus; ++i) {
                avg_curve.emplace_back(accu_cms[i].fpr(), accu_cms[i].tpr());
            }
            break;
        default:
            break;
    }
    return avg_curve;
}

vector<Point> micro_average_curve(const mat& P, const umat& T, EvalCentric centric, EvalMetric metric) {
    vector<double> weights;
    if (centric == EC_INS) {
        weights = vector<double>(P.n_cols, 1.0);
    } else {
        weights = vector<double>(P.n_rows, 1.0);
    }
    return micro_average_curve(P, T, centric, metric, weights);
}

double micro_average(const mat& P, const umat& T, EvalCentric centric, EvalMetric metric, const vector<double>& weights) {
    double res = 0.0;
    switch (metric) {
        case EM_FMAX:
            res = fmax(micro_average_curve(P, T, centric, EM_PRC));
            break;
        case EM_AUC:
            res = ordered_trapz(micro_average_curve(P, T, centric, EM_ROC));
            break;
        case EM_SDMIN:
            res = sdmin(micro_average_curve(P, T, centric, EM_RMC, weights));
            break;
        default:
            error_and_exit(__PRETTY_FUNCTION__, "Error metric.");
            break;
    }
    return res;
}

double micro_average(const mat& P, const umat& T, EvalCentric centric, EvalMetric metric) {
    vector<double> weights;
    if (centric == EC_INS) {
        weights = vector<double>(P.n_cols, 1.0);
    } else {
        weights = vector<double>(P.n_rows, 1.0);
    }
    return micro_average(P, T, centric, metric, weights);
}

// -------------
// Yuxiang Jiang (yuxjiang@indiana.edu)
// Department of Computer Science
// Indiana University, Bloomington
// Last modified: Wed 11 Sep 2019 02:09:03 AM E
