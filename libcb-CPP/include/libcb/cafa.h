#ifndef _LIBCB_CAFA_H_
#define _LIBCB_CAFA_H_

#include "util.h"
#include "ontology.h"
#include "data_matrix.h"

/**
 * @brief loads the prediction in CAFA format and outputs a consolidated
 * prediction data matrix
 */
dmat_t cafa_load(const Ontology& ont, const std::string& filename);

/**
 * @brief output predictions in CAFA format (without "header" and "footer")
 *
 * @remark this function assumes the all predictions are within the range [0,
 * 1].
 *
 * @param filename the output filename.
 * @param pred the predicted matrix.
 * @param seqs sequence identifiers.
 * @param terms term identifiers (e.g., GO term ID).
 * @param max_terms maximum number of terms per sequence, default = 1500.
 */
void cafa_save(
    const std::string& filename,
    const arma::mat& pred,
    const std::vector<std::string>& seqs,
    const std::vector<std::string>& terms,
    index_t max_terms = 1500);

/**
 * @brief output predictions in CAFA format (without "header" and "footer")
 *
 * @remark this function assumes the all predictions are within the range [0,
 * 1].
 *
 * @param filename the output filename.
 * @param dm the data matrix containing data, sequence & term tags
 * @param max_terms maximum number of terms per sequence, default = 1500.
 */
void cafa_save(
    const std::string& filename,
    const dmat_t& dm,
    index_t max_terms = 1500);

#endif // _LIBCB_CAFA_H_
