#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include "libcb/util.h"
#include "libcb/feature.h"

using std::cout;
using std::endl;
using std::vector;

/**
 * Allocate quota (i.e., number of samples) per sequence based on sequence
 * length and total quota (Q)
 */
void allocate_quota(const vector<index_t>& lens, index_t Q,
    vector<index_t>& quota) {
  if (Q == 0) {
    Q = std::numeric_limits<index_t>::max(); // 0 means unlimited
  }
  if (lens.size() > Q) {
    error_and_exit(__PRETTY_FUNCTION__, "Sample size should be larger than # of sequences.");
  }
  quota.resize(lens.size(), 1); // ensure >=1 sample per sequence
  index_t left = Q - lens.size();
  index_t L = std::accumulate(lens.begin(), lens.end(), 0) - lens.size();
  for (index_t i = 0; i < lens.size(); ++i) {
    index_t avail = static_cast<index_t>(std::floor(static_cast<double>(lens[i] - 1) / L * left));
    quota[i] += avail;
  }
}

void VQ::cluster(const vector<vector<double>>& data) {
  arma::arma_rng::set_seed_random();

  // allocate samples per sequence
  vector<index_t> lens, quota;
  for (const auto& seq : data) lens.push_back(seq.size());
  allocate_quota(lens, maxSamples, quota);

  index_t numberOfSamples = 0;
  for (index_t i = 0; i < lens.size(); ++i) {
    numberOfSamples += std::min(quota[i], (lens[i] - sampleLength + 1) / stepSize);
  }

  arma::mat samples(sampleLength, numberOfSamples);

  // reservoir sampling
  index_t cnt = 0;
  for (index_t i = 0; i < data.size(); ++i) {
    const auto& feat = data[i];
    if (feat.size() < sampleLength) {
      // Note that if this happens, the total number of samples will be less
      // than the desired count as the "quota" for this sequence is not used.
      warn(__PRETTY_FUNCTION__, "Segment length is longer than data length, skip");
      continue;
    } else {
      for (index_t startPosition = 0;
          startPosition + sampleLength < feat.size();
          startPosition += stepSize) {
        int target_pos = -1;
        if (cnt < quota[i]) {
          target_pos = cnt;
        } else {
          // generate a random number between 0 .. cnt - 1
          index_t index = arma::randi<index_t>(arma::distr_param(0, cnt - 1));
          if (index < quota[i]) {
            target_pos = index;
          }
        }
        cnt ++;
        if (target_pos >= 0) {
          // copy a sample
          for (index_t j = 0; j < sampleLength; ++j) {
            samples(j, target_pos) = feat[startPosition + j];
          }
        }
      }
    }
  }

  // call to kmeans to perform clustering
  arma::kmeans(centroids_, samples, numberOfClusters, arma::random_subset, 100, false);
}

void VQ::get_VQ_feature(const vector<double>& data,
    vector<double>& VQfeat,
    bool normalize) const {
  VQfeat = vector<double>(numberOfClusters, 0);
  // sliding window method
  for (index_t i = 0; i + sampleLength < data.size(); ++i) {
    // get a sample
    arma::vec x(sampleLength);
    for (index_t j = 0; j < sampleLength; ++j) {
      x(j) = data[i + j];
    }
    // find the nearest centroid
    index_t nearestIndex = 0;
    double smallestDist2 = std::numeric_limits<double>::max();
    for (index_t j = 0; j < centroids_.n_cols; ++j) {
      double dist2 = arma::sum(arma::square(centroids_.col(j) - x));
      if (smallestDist2 > dist2) {
        smallestDist2 = dist2;
        nearestIndex  = j;
      }
    }
    VQfeat[nearestIndex] += 1;
  }
  if (normalize) {
    double tot = std::accumulate(VQfeat.begin(), VQfeat.end(), 0);
    tot = tot < EPS ? 1.0 : tot;
    for (auto& vq : VQfeat) {
      vq /= tot;
    }
  }
}

void VQ::print_centroids() const {
  for (index_t i = 0; i < centroids_.n_cols; ++i) {
    cout << "CENTROID[" << i << "]:";
    centroids_.col(i).as_row().print();
  }
}

void VQ::serialize(std::ostream& stream) const {
  uint32_t data;

  data = maxSamples;
  SAVE_VAR(stream, data);

  data = sampleLength;
  SAVE_VAR(stream, data);

  data = stepSize;
  SAVE_VAR(stream, data);

  data = numberOfClusters;
  SAVE_VAR(stream, data);

  centroids_.save(stream);
}

void VQ::deserialize(std::istream& stream) {
  uint32_t data;

  LOAD_N(stream, data, sizeof(uint32_t));
  maxSamples = data;

  LOAD_N(stream, data, sizeof(uint32_t));
  sampleLength = data;

  LOAD_N(stream, data, sizeof(uint32_t));
  stepSize = data;

  LOAD_N(stream, data, sizeof(uint32_t));
  numberOfClusters = data;

  centroids_.load(stream);
}
