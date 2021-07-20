//! libcb/util.h

#ifndef _LIBCB_UTIL_H_
#define _LIBCB_UTIL_H_

#include <algorithm>
#include <chrono>
#include <ctime>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// #include <boost/filesystem.hpp> // provide functionality of has_file()
#include <unistd.h>
#include <armadillo>

// Test if one item belong to a container.
// Note that this is for convenience, but use it with caution.
#define CONTAINS(c, x) (c.find(x) != c.end())

// save and load
#define LOAD_N(s, b, n) (s.read((char*)(&b), n))
#define SAVE_VAR(s, b) (s.write(reinterpret_cast<const char*>(&b), sizeof(decltype(b))))

// for convenience usage of armadillo
#define EPS arma::datum::eps
#define INF arma::datum::inf

// adopting armadillo's convention for indexing type (see its documentation)
typedef arma::uword index_t;

/**
 * @brief Prints a warning message to stderr.
 *
 * @param func a function name.
 * @param msg a message.
 */
inline void warn(const std::string& func, const std::string& msg) {
  std::cerr
    << "[WARNING] in " << func << ": "
    << msg << std::endl;
}

/**
 * @brief Prints an error message to stderr and exit the program.
 *
 * @param func a function name.
 * @param msg a message.
 */
inline void error_and_exit(const std::string& func, const std::string& msg) {
  std::cerr
    << "[ERROR] in " << func << ": "
    << msg << std::endl;
  exit(1);
}

/**
 * @brief Prints a logging message to stderr.
 *
 * @param func a function name.
 * @param msg a message.
 */
inline void log_message(const std::string& func, const std::string& msg) {
  using namespace std::chrono;
  time_t t = system_clock::to_time_t(system_clock::now());
  std::string timestr = ctime(&t);
  // remove the ending '\n'
  timestr = timestr.substr(0, timestr.length() - 1);
  std::cerr
    << "[" << timestr << "] in " << func << ": "
    << msg << std::endl;
}

/**
 * @class SparseTable
 *
 * @brief a class of @a sparse table.
 *
 * @remark It is required that both `rid_type` and `cid_type` have their @a
 * less-than operator defined. In most cases, one can use `std::string` as these
 * ID types.
 */
template<typename rid_type, typename cid_type, typename T>
class SparseTable {
  public:
    //! Type of size
    using size_type = index_t;

    //! Type of data
    using data_type = T;

  protected:
    //! The set of rows.
    std::set<rid_type> m_row;

    //! The set of columns.
    std::set<cid_type> m_col;

    //! The row-wise elements.
    std::unordered_map<rid_type, std::unordered_map<cid_type, T>> m_row_elem;

    //! The column-wise elements.
    std::unordered_map<cid_type, std::unordered_map<rid_type, T>> m_col_elem;

  public:
    /**
     * @brief Gets the number of rows.
     *
     * @return The number of rows.
     *
     * @sa number_of_cols()
     */
    size_type number_of_rows() const {
      return m_row.size();
    }

    /**
     * @brief Gets the number of columns.
     *
     * @return The number of columns.
     *
     * @sa number_of_rows()
     */
    size_type number_of_cols() const {
      return m_col.size();
    }

    /**
     * @brief Gets the number of non-zero elements.
     *
     * @return The number of non-zero elements.
     */
    size_type number_of_elements() const {
      size_type num = 0;
      for (const auto& r : m_row_elem) {
        num += r.second.size();
      }
      return num;
    }

    /**
     * @brief The assign operator.
     *
     * @param t another SparseTable object.
     *
     * @return A constant reference to itself.
     */
    const SparseTable& operator=(const SparseTable& t) {
      if (this != &t) {
        m_row      = t.m_row;
        m_col      = t.m_col;
        m_row_elem = t.m_row_elem;
        m_col_elem = t.m_col_elem;
      }
      return *this;
    }

    /**
     * @brief Test if a row exists.
     *
     * @param row A row ID.
     *
     * @return True or False.
     *
     * @sa has_col()
     */
    bool has_row(const rid_type& row) const {
      return CONTAINS(m_row, row);
    }

    /**
     * @brief Test if a column exists.
     *
     * @param col A column ID.
     *
     * @return True or False.
     *
     * @sa has_row()
     */
    bool has_col(const cid_type& col) const {
      return CONTAINS(m_col, col);
    }

    /**
     * @brief Adds a row.
     *
     * @param row a row ID.
     *
     * @sa add_col()
     */
    void add_row(const rid_type& row) {
      if (!CONTAINS(m_row, row)) {
        m_row.insert(row);
        std::unordered_map<cid_type, T> empty_map;
        m_row_elem.insert(make_pair(row, empty_map));
      }
    }

    /**
     * @brief Adds a column.
     *
     * @param col a column ID.
     *
     * @sa add_row()
     */
    void add_col(const cid_type& col) {
      if (!CONTAINS(m_col, col)) {
        m_col.insert(col);
        std::unordered_map<rid_type, T> empty_map;
        m_col_elem.insert(make_pair(col, empty_map));
      }
    }

    /**
     * @brief Adds an element with row and column ID.
     *
     * @param row a row ID.
     * @param col a column ID.
     * @param v a value of type data_type.
     *
     * @sa remove_element()
     */
    void add_element(
        const rid_type& row,
        const cid_type& col,
        const T& v) {
      add_row(row);
      add_col(col);
      m_row_elem[row][col] = m_col_elem[col][row] = v;
    }

    /**
     * @brief Removes an element from the table.
     *
     * @param row a row ID.
     * @param col a column ID.
     *
     * @sa add_element()
     */
    void remove_element(
        const rid_type& row,
        const cid_type& col) {
      if (CONTAINS(m_row, row)) {
        m_row_elem[row].erase(col);
      }
      if (CONTAINS(m_col, col)) {
        m_col_elem[col].erase(row);
      }
    }

    /**
     * @brief Removes an entire row.
     *
     * @param row a row ID.
     *
     * @sa remove_col()
     */
    void remove_row(const rid_type& row) {
      if (CONTAINS(m_row, row)) {
        m_row.erase(row);
        for (const auto& c : m_row_elem[row]) {
          m_col_elem[c.first].erase(row);
        }
        m_row_elem.erase(row);
      }
    }

    /**
     * @brief Removes an entire column.
     *
     * @param col a column ID.
     *
     * @sa remove_row()
     */
    void remove_col(const cid_type& col) {
      if (CONTAINS(m_col, col)) {
        m_col.erase(col);
        for (const auto& r : m_col_elem[col]) {
          m_row_elem[r.first].erase(col);
        }
        m_col_elem.erase(col);
      }
    }

    /**
     * @brief Gets the full matrix representation
     *
     * @return A tuple of of <matrix, row IDs, col IDs>
     */
    std::tuple<std::vector<std::vector<data_type>>, std::vector<rid_type>, std::vector<cid_type>> full() const {
      using namespace std;
      vector<rid_type> rows;
      vector<cid_type> cols;
      for (const auto& r : m_row) {
        rows.push_back(r);
      }
      for (const auto& c : m_col) {
        cols.push_back(c);
      }
      vector<vector<data_type>> matrix(m_row.size(), vector<data_type>(m_col.size()));
      for (index_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        if (CONTAINS(m_row_elem, r)) {
          for (index_t j = 0; j < cols.size(); ++j) {
            const auto& c = cols[j];
            if (CONTAINS(m_row_elem.at(r), c)) {
              matrix[i][j] = m_row_elem.at(r).at(c);
            }
          }
        }
      }
      return make_tuple(matrix, rows, cols);
    }
};

/**
 * @brief Loads a vector of type T from a file.
 *
 * @pre This function requires operator>> of T is properly defined.
 *
 * @tparam T
 * @param ifile an input file name.
 *
 * @return A vector of loaded items of type T.
 */
template<typename T>
std::vector<T> load_items(const std::string& ifile) {
  std::ifstream ifs(ifile, std::ifstream::in);
  std::vector<T> items;
  T item;
  while (ifs >> item) {
    items.push_back(item);
  }
  ifs.close();
  return items;
}

/**
 * @brief Returns the union of two vectors of type T.
 *
 * @pre This function requires the "less-than" operator of type T is properly
 * defined and also T should be hashable.
 *
 * @param A a vector of type T.
 * @param B another vector of type T.
 * @param ia the index of the union from A. I.e., A[i] is the ia[i]-th element
 * in the union.
 * @param ib the index of the union from B. I.e., B[i] is the ib[i]-th element
 * in the union.
 *
 * @return The union of type T.
 */
template<typename T>
std::vector<T> vector_union(const std::vector<T>& A, const std::vector<T>& B, std::vector<index_t>& ia, std::vector<index_t>& ib) {
  std::set<T> S;
  std::vector<T> U;
  std::unordered_map<T, index_t> T2i;
  S.insert(A.begin(), A.end());
  S.insert(B.begin(), B.end());
  index_t index = 0;
  for (const auto& elem : S) {
    T2i[elem] = index++;
    U.push_back(elem);
  }
  ia.resize(A.size());
  for (size_t i = 0; i < A.size(); ++i) {
    ia[i] = T2i.at(A[i]);
  }
  ib.resize(B.size());
  for (size_t i = 0; i < B.size(); ++i) {
    ib[i] = T2i.at(B[i]);
  }
  return U;
}

inline
bool has_file(const std::string& filename) {
  return access(filename.c_str(), F_OK) != -1;
}

/**
 * @brief Parses a configuration file into pairs of strings (Name, Value).
 *
 * @remark The format of configuration file follows: [name] = [value]
 * @remark It allows the comment sign '#'
 * @remark Do NOT use single/double quotes around values.
 * @remark All white spaces including TAB, and SPACE are ignored.
 *
 * @param cfile a configuration file.
 */
std::unordered_map<std::string, std::string> parse_config(const std::string&);

/**
 * @brief Trims whitespaces from both ends of a string.
 *
 * @param s a string.
 * @param whitespaces A string of "white spaces" to trim.
 *
 * @return A resulting string.
 */
std::string trim(const std::string& s, const std::string& whitespaces = " \t\n\v\f\r");

/**
 * @brief Returns the specified quantile.
 *
 * @remark This function performs "linear interpolation" if exact quantile is
 * not available. It emulates the behavior of MATLAB's `quantile` function.
 *
 * @tparam T must be floating point types, e.g., float, double.
 * @param x a vector of type T.
 * @param p a cumulative probability.
 * @param sorted indicator of the given x is sorted or not
 *
 * @return The corresponding quantile.
 */
template<typename T = double>
T quantile(const std::vector<T>& x, double p, bool sorted = false) {
  if (p < 0 || p > 1) {
    error_and_exit(__PRETTY_FUNCTION__, "Probability must in [0, 1].");
  }
  std::vector<T> z = x; // make a local copy
  size_t n = z.size();
  if (!sorted) {
      sort(z.begin(), z.end());
  }
  // data position: |   0   |   1   | ... |    n-1    |
  // quantiles:     | 0.5/n | 1.5/n | ... | (n-0.5)/n |
  index_t posl = static_cast<index_t>(std::max(0.0, std::floor(p * n - 0.5)));
  index_t posr = static_cast<index_t>(std::min(n - 1.0, std::ceil(p * n - 0.5)));
  if (posl == posr) {
    return z[posl];
  }
  double pl = (static_cast<double>(posl) + 0.5) / n;
  double pr = (static_cast<double>(posr) + 0.5) / n;
  return z[posl] + (p - pl) / (pr - pl) * (z[posr] - z[posl]); // linear interpolation
}

/**
 * @brief random sample from n objects with or without replacement, indices of sampled elements will
 * be returned.
 *
 * @remark this function is similar to its MATLAB counterpart, but shift by one in index (i.e.,
 * starting from 0 rather than 1).
 *
 * @param n the size of population.
 * @param k the size of sample.
 * @param replacement with or without replacement, default false.
 *
 * @return an Armadillo vector of sampled index between 0 and n-1.
 */
arma::uvec randsample(index_t n, index_t k, bool replacement = false);

/**
 * @brief random sample from a given population.
 *
 * @tparam T data type, default: double.
 * @param population the population, as an Armadillo column object.
 * @param k the size of sample.
 * @param replacement with or without replacement, default false.
 *
 * @return a sample.
 */
template<class T = double>
arma::Col<T> randsample(const arma::Col<T>& population, index_t k, bool replacement = false) {
  arma::uvec indices = randsample(population.n_elem, k, replacement);
  return population(indices);
}

#endif // _LIBCB_UTIL_H_

// -------------
// Yuxiang Jiang (yuxjiang@indiana.edu)
// Department of Computer Science
// Indiana University, Bloomington
// Last modified: Wed 25 Sep 2019 09:28:35 PM P
