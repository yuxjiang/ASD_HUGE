#ifndef _LIBCB_FEATURE_H_
#define _LIBCB_FEATURE_H_

#include <fstream>
#include <vector>
#include <armadillo>

#include "util.h"

/**
 * class: Vector Quantization (VQ)
 */
class VQ {
  public:
    VQ() :
      maxSamples(0),
      sampleLength(16),
      stepSize(1),
      numberOfClusters(16) {}

    void cluster(const std::vector<std::vector<double>>& data);

    /**
     * Get VQ feature from a sequential data vector
     *
     * @param data, a sequential data of variable length
     * @param VQfeat, a fixed length VQ feature vector
     */
    void get_VQ_feature(
        const std::vector<double>& data,
        std::vector<double>& VQfeat,
        bool normalize
        ) const;

    void print_centroids() const;

    void serialize(std::ostream& stream) const;

    void deserialize(std::istream& stream);

  public:
    /**
     * The maximum number of samples to extract from raw data and used for
     * clustering, note that this number should be larger than the number of
     * sequences as we need as least one sample from each sequence. Also, the
     * actual number of sampled could be smaller.
     */
    index_t maxSamples;

    /**
     * The length of each sample, aka. "window size"
     */
    index_t sampleLength;

    /**
     * The skip step between two samples from a sequence
     * e.g. step = 1 means the 2nd sample is right next to the 1st sample.
     */
    index_t stepSize;

    /**
     * The number of cluters for running k-means clustering. This will be also
     * the length of VQ features.
     */
    index_t numberOfClusters;

  private:
    /**
     * Each centroid will be stored as a column vector
     */
    arma::mat centroids_;
};

#endif // _LIBCB_FEATURE_H_
