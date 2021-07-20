//! libcb/annotation.h

#ifndef _LIBCB_ANNOTATION_H_
#define _LIBCB_ANNOTATION_H_

#include "util.h"
#include "data_matrix.h"
#include "ontology.h"

namespace gaf {
  /**
   * @brief GAF2.0 qualifiers.
   */
  typedef enum {
    unknown = 0,      //!< default value
    positive,         //!< positive annotation
    NOT,              //!< negative annotation
    colocalizes_with, //!< not used
    contributes_to    //!< not used
  } qualifier_t;
}

/**
 * @class AnnotationTable
 *
 * @brief a class of @a annotation @a table.
 *
 * @remark It stores association pairs as (sequence, term), both are represented
 * as `std::string`. Association status has the `gaf::qualifier_t` type,
 * where empty value indicates @a unknown.
 */
class AnnotationTable: public SparseTable<std::string, std::string, gaf::qualifier_t> {
  public:
    //! Type of sequence ID
    using sid_type = std::string; // to denote sequence ID

    //! Type of term ID
    using tid_type = std::string; // to denote term ID

  public:
    /**
     * @brief Loads annotations with a fixed ontology.
     *
     *  A parsed annotation file has a @a TWO-COLUMN format delimited by TAB as:
     *  `[sequence ID] [term ID]`, indicates only positive annotations.
     *
     * @param afiles a vector of parsed annotation files.
     * @param ont an Ontology object.
     */
    void load_parsed_gaf_with_ontology(const std::vector<std::string>& afiles, const Ontology& ont);

    /**
     * @brief Gets the set of annotated terms of a sequence.
     *
     * @param sid a sequence ID.
     *
     * @return The set of term IDs.
     *
     * @sa annotated_sequences_of()
     */
    std::set<std::string> annotated_terms_of(const sid_type& sid) const {
      std::set<std::string> res;
      if (has_row(sid)) {
        for (const auto& t : m_row_elem.at(sid)) {
          res.insert(t.first);
        }
      }
      return res;
    }

    /**
     * @brief Gets the set of annotated sequences of a term.
     *
     * @param term a term ID.
     *
     * @return The set of sequence IDs.
     *
     * @sa annotated_terms_of()
     */
    std::set<std::string> annotated_sequences_of(const tid_type& term) const {
      std::set<std::string> res;
      if (has_col(term)) {
        for (const auto& s : m_col_elem.at(term)) {
          res.insert(s.first);
        }
      }
      return res;
    }

    /**
     * @brief Transfers annotated terms from one sequence to another.
     *
     * @param src the source sequence ID.
     * @param dst the destination sequence ID.
     *
     * @sa transfer_annotated_terms()
     */
    void transfer_annotated_terms(const sid_type& src, const sid_type& dst) {
      if (!has_row(src)) {
        error_and_exit(
            "AnnotationTable::transfer_annotated_terms()",
            std::string("source sequence does not exist: ") + src
            );
      }
      for (const auto& t : m_row_elem.at(src)) {
        add_element(dst, t.first, t.second);
      }
    }

    /**
     * @brief Transfers annotated terms from one sequence to another.
     *
     * @param src the source term ID.
     * @param dst the destination term ID.
     *
     * @remark both src and dst terms must be present in the ontology.
     *
     * @sa transfer_annotated_sequences()
     */
    void transfer_annotated_sequences(const tid_type& src, const tid_type& dst) {
      if (!has_col(src)) {
        error_and_exit(
            "AnnotationTable::transfer_annotated_sequences()",
            std::string("source term does not exist: ") + src
            );
      }
      if (!has_col(dst)) {
        error_and_exit(
            "AnnotationTable::transfer_annotated_sequences()",
            std::string("destination term does not exist: ") + dst
            );
      }
      for (const auto& s : m_col_elem.at(src)) {
        add_element(s.first, dst, s.second);
      }
    }
};

/**
 * @brief a class of annotation.
 */
class Annotation {
  private:
    //! The asssociated ontology structure.
    Ontology m_ont;

    //! The annotation table.
    AnnotationTable m_a;

    //! A boolean indicator of whether the annotation is replenished.
    bool m_replenished;

  public:
    //! The sequence type.
    using sid_type = AnnotationTable::sid_type;

    //! The term type.
    using tid_type = AnnotationTable::tid_type;

  public:
    //! The default constructor
    Annotation() {};

    /**
     * @brief A constructor from an OBO file and an annotation file.
     *
     * @param obofile an OBO file.
     * @param afile a @a TWO-COLUMN annotation file.
     */
    Annotation(const std::string& obofile, const std::string& afile) {
      m_ont = Ontology(obofile);
      _load_afiles(std::vector<std::string>(1, afile));
      replenish();
    }

    /**
     * @brief A constructor from an OBO file and multiple annotation files.
     *
     * @param obofile an OBO file.
     * @param afiles a vector of parsed @a TWO-COLUMN annotation files.
     */
    Annotation(const std::string& obofile, const std::vector<std::string>& afiles) {
      m_ont = Ontology(obofile);
      _load_afiles(afiles);
      replenish();
    }

    /**
     * @brief A constructor from an Ontology object and an annotation file.
     *
     * @param ont an Ontology object.
     * @param afile a @a TWO-COLUMN annotation file.
     */
    Annotation(const Ontology& ont, const std::string& afile) {
      m_ont = ont;
      _load_afiles(std::vector<std::string>(1, afile));
      replenish();
    }

    /**
     * @brief A constructor from an Ontology object and multiple annotation
     * files.
     *
     * @param ont an Ontoloy object.
     * @param afiles a vector of @a TWO-COLUMN annotation files.
     */
    Annotation(const Ontology& ont, const std::vector<std::string>& afiles) {
      m_ont = ont;
      _load_afiles(afiles);
      replenish();
    }

    /**
     * @brief A copy constructor.
     *
     * @param anno another Annotation object.
     */
    Annotation(const Annotation& anno) {
      m_ont = anno.m_ont;
      m_a   = anno.m_a;
      m_replenished = anno.m_replenished;
    }

    /**
     * @brief An assign operator.
     *
     * @param anno another Annotation object.
     *
     * @return A constant reference to itself.
     */
    const Annotation& operator=(const Annotation& anno) {
      if (this != &anno) {
        m_ont = anno.m_ont;
        m_a   = anno.m_a;
        m_replenished = anno.m_replenished;
      }
      return (*this);
    }

    inline index_t number_of_sequences() const {
      return m_a.number_of_rows();
    }

    inline index_t number_of_terms() const {
      return m_a.number_of_cols();
    }

    /**
     * @brief Returns the number of positive annotations.
     *
     * @return The number of positive annotations.
     */
    inline index_t number_of_annotations() const {
      return m_a.number_of_elements();
    }

    /**
     * @brief Returns an integer matrix (only 0 and 1 are valid values)
     * prepresenting the annotation all sequences
     */
    DataMatrix<index_t> binary_data_matrix() const {
      std::vector<std::vector<gaf::qualifier_t>> anno;
      std::vector<std::string> seqs, terms;
      std::tie(anno, seqs, terms) = m_a.full();
      arma::umat matrix(seqs.size(), terms.size(), arma::fill::zeros);
      for (index_t i = 0; i < seqs.size(); ++i) {
        for (index_t j = 0; j < terms.size(); ++j) {
          if (anno[i][j] == gaf::positive ||
              anno[i][j] == gaf::colocalizes_with ||
              anno[i][j] == gaf::contributes_to) {
            matrix(i, j) = 1;
          }
        }
      }
      return DataMatrix<index_t>(matrix, seqs, terms);
    }

    /**
     * @brief Gets the annotated terms of a sequence.
     *
     * @param seq a sequence ID.
     *
     * @return A set of term IDs.
     */
    std::set<std::string> annotated_terms_of(const sid_type& seq) const {
      return m_a.annotated_terms_of(seq);
    }

    /**
     * @brief Gets the annotated sequences of a term.
     *
     * @param term a term ID.
     *
     * @return A set of sequence IDs.
     */
    std::set<std::string> annotated_sequences_of(const tid_type& term) const {
      return m_a.annotated_sequences_of(term);
    }

    /**
     * @brief Adds an association between a pair of sequence and term.
     *
     * @param seq a sequence ID.
     * @param term a term ID.
     * @param q a qualifier of type `gaf::qualifier_t`.
     */
    void add_annotation(const sid_type& seq, const tid_type& term,
        gaf::qualifier_t q = gaf::positive) {
      m_a.add_element(seq, term, q);
    }

    /**
     * @brief Removes an annotation.
     *
     * @param seq a sequence ID.
     * @param term a term ID.
     */
    void remove_annotation(const sid_type& seq, const tid_type& term) {
      m_a.remove_element(seq, term);
    }

    /**
     * @brief Removes sequences with ONLY @a uninformative "protein binding"
     * annotation.
     *
     * @remark @a Unformative "protein binding" means a GO:0005515 annotation
     * as a sequence's only annotated term.
     */
    void remove_uninformative_protein_binding();

    /**
     * @brief Replenishes annotation up to the root(s).
     *
     * @sa deplete()
     */
    void replenish();

    /**
     * @brief Depletes annotation down to keep only most informative the
     * term(s).
     *
     * @sa replenish()
     */
    void deplete();

  protected:
    /**
     * @brief Reads annotation files.
     *
     * @param afiles a vector of @a TWO-COLUMN annotation files.
     */
    void _load_afiles(const std::vector<std::string>& afiles) {
      m_a.load_parsed_gaf_with_ontology(afiles, m_ont);
    }
};

#endif // _LIBCB_ANNOTATION_H_

// end of file

// -------------
// Yuxiang Jiang (yuxjiang@indiana.edu)
// Department of Computer Science
// Indiana University, Bloomington
// Last modified: Wed 25 Sep 2019 12:01:05 AM P
