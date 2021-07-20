#ifndef _LIBCB_DATA_MATRIX_H_
#define _LIBCB_DATA_MATRIX_H_

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <utility>

#include "util.h" // error_and_exit()

/**
 * @brief class DataMatrix.
 *
 * @remark this class is designed for holding an n-by-m data matrix with both rows and columns
 * attaching a std::string tag.
 *
 * @remark this class depends on the Armadillo C++ library, so that each element should have a type
 * supported by that library. By default: double.
 */
template<typename T>
struct DataMatrix {
  using data_type  = T;

  std::vector<std::string> row_tag; // tags for each row (data points)
  std::vector<std::string> col_tag; // tags for each col (features)
  arma::Mat<data_type> data;    // n-by-m data matrix

  DataMatrix() {}

  /**
   * @brief the basic constructor.
   *
   * @param data_ data matrix (from Armadillo)
   * @param row_tag_ row tags.
   * @param col_tag_ column tags.
   */
  DataMatrix(const arma::Mat<data_type>& data_, const std::vector<std::string>& row_tag_,
      const std::vector<std::string>& col_tag_) {
    if (row_tag_.size() != data_.n_rows) {
      error_and_exit( __PRETTY_FUNCTION__, "Inconsistent number of rows.");
    }
    if (col_tag_.size() != data_.n_cols) {
      error_and_exit( __PRETTY_FUNCTION__, "Inconsistent number of columns.");
    }
    row_tag = row_tag_;
    col_tag = col_tag_;
    data    = data_;
  }

  /**
   * @brief the constructor without tags
   *
   * @param data_ data matrix (from Armadillo)
   */
  DataMatrix(const arma::Mat<data_type>& data_) {
    data = data_;
    // default tag (for both rows and cols): 0, 1, 2, ...
    row_tag = std::vector<std::string>(data_.n_rows);
    col_tag = std::vector<std::string>(data_.n_cols);
    for (size_t i = 0; i < row_tag.size(); ++i) {
      row_tag[i] = std::to_string(i);
    }
    for (size_t i = 0; i < col_tag.size(); ++i) {
      col_tag[i] = std::to_string(i);
    }
  }

  /**
   * @brief the constructor without data, only provide tags to specify its dimension. Each entry
   * will be initialized as zero.
   *
   * @param row_tag_ the row tags
   * @param col_tag_ the column tage
   */
  DataMatrix(const std::vector<std::string> row_tag_, const std::vector<std::string> col_tag_) {
    row_tag = row_tag_;
    col_tag = col_tag_;
    data = arma::Mat<data_type>(row_tag_.size(), col_tag_.size(), arma::fill::zeros);
  }

  /**
   * @brief the copy constructor.
   *
   * @param other another DataMatrix object.
   */
  DataMatrix(const DataMatrix& other) : row_tag(other.row_tag), col_tag(other.col_tag),
    data(other.data) {}

  /**
   * @brief the move constructor.
   *
   * @param other another DataMatrix object.
   */
  DataMatrix(DataMatrix&& other) : row_tag(std::move(other.row_tag)),
    col_tag(std::move(other.col_tag)),
    data(std::move(other.data)) {}

  ~DataMatrix() = default;

  /**
   * @brief the assign operator.
   *
   * @param other a data matrix object.
   *
   * @return a reference to this object.
   */
  const DataMatrix& operator=(const DataMatrix& other) {
    if (this != &other) {
      row_tag = other.row_tag;
      col_tag = other.col_tag;
      data    = other.data;
    }
    return *this;
  }

  /**
   * @brief an extra "assign" operator with Armadillo matrix.
   *
   * @caveat this function is provided for convenience. However, default row/column tags will be
   * assigned.
   *
   * @param dm an Armadillo matrix object.
   *
   * @return a reference to this object.
   */
  const DataMatrix& operator=(const arma::Mat<data_type>& dm) {
    data = dm;
    row_tag = std::vector<std::string>(dm.n_rows);
    col_tag = std::vector<std::string>(dm.n_cols);
    // set row/column tag to 1, 2, ...
    for (size_t i = 0; i < row_tag.size(); ++i) {
      row_tag[i] = std::to_string(i + 1);
    }
    for (size_t i = 0; i < col_tag.size(); ++i) {
      col_tag[i] = std::to_string(i + 1);
    }
    return *this;
  }

  inline index_t num_rows() const { return data.n_rows; }

  inline index_t num_cols() const { return data.n_cols; }

  /**
   * @brief check for empty matrix.
   *
   * @return True or false.
   */
  inline bool empty() const { return data.is_empty(); }

  /**
   * @brief returns a subset (a collection of rows) of the origin DataMatrix object.
   *
   * @param indices the index of selected rows.
   *
   * @return the subset DataMatrix object.
   */
  DataMatrix<data_type> subset(const arma::uvec& indices) const {
    if (indices.max() >= data.n_rows) {
      error_and_exit(__PRETTY_FUNCTION__, "Index out of range.");
    }
    std::vector<std::string> rtags(indices.n_elem);
    for (index_t i = 0; i < indices.n_elem; ++i) {
      rtags[i] = row_tag[indices[i]];
    }
    return DataMatrix<data_type>(data.rows(indices), rtags, col_tag);
  }

  /**
   * @brief returns a subset (a collection of columns) of the origin DataMatrix object.
   *
   * @param indices the index of selected columns.
   *
   * @return the subset DataMatrix object.
   */
  DataMatrix<data_type> subset(const std::vector<index_t>& indices) const {
    return subset(arma::uvec(indices));
  }

  /**
   * @brief projects the data matrix onto a selection of rows and keep all columns.
   *
   * @param rt a vector of row tags.
   * @return the projected data matrix.
   */
  DataMatrix<data_type> project(const std::vector<std::string>& rt) const {
    return project(rt, col_tag);
  }

  /**
   * @brief projects the data matrix onto a selection of rows and columns.
   *
   * @param rt a vector of row tags.
   * @param ct a vector of column tags.
   * @return the projected data matrix.
   */
  DataMatrix<data_type> project(const std::vector<std::string>& rt,
      const std::vector<std::string>& ct) const {
    DataMatrix<data_type> dm(rt, ct);
    // build up indices
    std::unordered_map<std::string, index_t> row_index, col_index;
    for (index_t i = 0; i < row_tag.size(); ++i) {
      row_index[row_tag[i]] = i;
    }
    for (index_t i = 0; i < col_tag.size(); ++i) {
      col_index[col_tag[i]] = i;
    }
    for (index_t i = 0; i < rt.size(); ++i) {
      for (index_t j = 0; j < ct.size(); ++j) {
        if (CONTAINS(row_index, rt[i]) && CONTAINS(col_index, ct[j])) {
          dm.data(i, j) = data(row_index.at(rt[i]), col_index.at(ct[j]));
        }
      }
    }
    return dm;
  }

  void serialize(std::ostream& stream) const {
    uint32_t val;
    val = data.n_rows;
    SAVE_VAR(stream, val);
    for (index_t i = 0; i < data.n_rows; ++i) {
      std::string id = row_tag[i];
      uint32_t len = id.length();
      SAVE_VAR(stream, len);
      stream.write(id.c_str(), len);
    }
    val = data.n_cols;
    SAVE_VAR(stream, val);
    for (index_t i = 0; i < data.n_cols; ++i) {
      std::string id = col_tag[i];
      uint32_t len = id.length();
      SAVE_VAR(stream, len);
      stream.write(id.c_str(), len);
    }
    data.save(stream);
  }

  void deserialize(std::istream& stream) {
    uint32_t n, val;
    char buf[256];
    LOAD_N(stream, n, sizeof(uint32_t));
    row_tag.resize(n);
    for (index_t i = 0; i < n; ++i) {
      LOAD_N(stream, val, sizeof(uint32_t));
      LOAD_N(stream, buf, val);
      std::string id(buf, val);
      row_tag[i] = id;
    }
    LOAD_N(stream, n, sizeof(uint32_t));
    col_tag.resize(n);
    for (index_t i = 0; i < n; ++i) {
      LOAD_N(stream, val, sizeof(uint32_t));
      LOAD_N(stream, buf, val);
      std::string id(buf, val);
      col_tag[i] = id;
    }
    data.load(stream);
  }

  /**
   * @brief Saves the data matrix into a CSV file
   *
   * @param file is the output file name.
   * @param prec is the output precision, default to 6
   */
  void save_csv(const std::string& file, int prec=6) const {
    std::ofstream ofs;
    ofs.precision(prec);
    ofs.open(file, std::ofstream::out);
    // output header line
    ofs << "id";
    for (index_t i = 0; i < data.n_cols; ++i) {
      ofs << "," << col_tag[i];
    }
    ofs << "\n";
    for (index_t i = 0; i < data.n_rows; ++i) {
      ofs << row_tag[i];
      for (index_t j = 0; j < data.n_cols; ++j) {
        ofs << "," << std::scientific << data(i, j);
      }
      ofs << "\n";
    }
    ofs.close();
  }
};

template<typename X, typename Y>
struct DataSet {
  using x_type = X;
  using y_type = Y;

  // x and y should always have the same number of rows
  DataMatrix<x_type> x;
  DataMatrix<y_type> y;

  DataSet() {};

  /**
   * @brief the constructor with two DataMatrix objects x and y.
   *
   * @param x_ the feature matrix x.
   * @param y_ the labelling matrix y.
   */
  DataSet(const DataMatrix<x_type>& x_, const DataMatrix<y_type>& y_) {
    if (x_.num_rows() != y_.num_rows()) {
      error_and_exit( __PRETTY_FUNCTION__,
          "Inconsistent number of rows: x(" +
          std::to_string(x_.num_rows()) +
          "), y(" + std::to_string(y_.num_rows()) +
          ")\n");
    }
    x = x_;
    y = y_;
  }

  /**
   * @brief the copy constructor.
   *
   * @param other another DataSet object.
   */
  DataSet(const DataSet& other) : x(other.x), y(other.y) {}

  /**
   * @brief the move constructor.
   *
   * @param other another DataSet object.
   */
  DataSet(DataSet&& other) : x(std::move(other.x)), y(std::move(other.y)) {}

  inline index_t num_examples() const {
    return x.num_rows();
  }

  inline index_t num_features() const {
    return x.num_cols();
  }

  inline index_t num_targets() const {
    return y.num_cols();
  }

  inline DataSet<x_type, y_type> subset(const arma::uvec& indices) const {
    return DataSet<x_type, y_type>(x.subset(indices), y.subset(indices));
  }

  inline DataSet<x_type, y_type> subset(const std::vector<index_t>& indices) const {
    return subset(arma::uvec(indices));
  }

  /**
   * @brief random two-way partition.
   *
   * @param p0 proportion of set 0.
   * @param p1 proportion of set 1.
   *
   * @return a collection of partitioned DataSet objects.
   */
  std::pair<std::vector<DataSet<x_type, y_type>>, std::vector<arma::uvec>>
    partition(double p0, double p1) const {
      std::vector<double> proportion = {p0, p1};
      return partition(proportion);
    }

  /**
   * @brief random three-way partition.
   *
   * @param p0 proportion of set 0.
   * @param p1 proportion of set 1.
   * @param p2 proportion of set 2.
   *
   * @return a collection of partitioned DataSet objects.
   */
  std::pair<std::vector<DataSet<x_type, y_type>>, std::vector<arma::uvec>>
    partition(double p0, double p1, double p2) const {
      std::vector<double> proportion = {p0, p1, p2};
      return partition(proportion);
    }

  /**
   * @brief random n-way partition.
   *
   * @param proportion proportions of sets.
   *
   * @return a collection of partitioned DataSet objects.
   */
  std::pair<std::vector<DataSet<x_type, y_type>>, std::vector<arma::uvec>>
    partition(const std::vector<double>& proportion) const {
      size_t k = proportion.size();
      index_t n = x.num_rows();
      double sum = 0.0;
      std::vector<double> cumsum;
      std::vector<index_t> bins;
      cumsum.push_back(sum);
      for (size_t i = 0; i < k; ++i) {
        sum += proportion[i];
        cumsum.push_back(sum);
      }
      // normalized cumsum
      // bins has (n+1) elements, corresp. to n slots
      // the i-th partition spans: [bins[i-1], bins[i]), i = 1, ..., n
      for (size_t i = 0; i < cumsum.size(); ++i) {
        cumsum[i] /= sum;
        bins.push_back(static_cast<index_t>(cumsum[i] * n));
      }
      // shuffle index 0, 1, ..., n - 1
      arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0, n - 1, n));
      // build subsets as partitions
      std::vector<DataSet<x_type, y_type>> ds_splits;
      std::vector<arma::uvec> indices_splits;
      for (size_t i = 0; i < proportion.size(); ++i) {
        if (bins[i] < bins[i + 1]) {
          arma::uvec indices_split_i = indices(arma::span(bins[i], bins[i + 1] - 1));
          ds_splits.emplace_back(x.subset(indices_split_i), y.subset(indices_split_i));
          indices_splits.emplace_back(indices_split_i);
        } else {
          // empty partition
          ds_splits.emplace_back(DataMatrix<x_type>(), DataMatrix<y_type>());
          indices_splits.emplace_back(arma::uvec());
        }
      }
      return std::make_pair(ds_splits, indices_splits);
    }
};

typedef DataMatrix<double>      dmat_t;
typedef DataSet<double, double> dset_t;

#endif // _LIBCB_DATA_MATRIX_H_

// -------------
// Yuxiang Jiang (yuxjiang@indiana.edu)
// Department of Computer Science
// Indiana University Bloomington
// Last modified: Mon 07 Oct 2019 10:11:28 PM P
