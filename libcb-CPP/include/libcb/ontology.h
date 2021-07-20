//! libcb/ontology.h

#ifndef _LIBCB_ONTOLOGY_H_
#define _LIBCB_ONTOLOGY_H_

#include "util.h"
#include <armadillo>

/**
 * @class Ontology
 *
 * @brief a class of ontology.
 */
class Ontology {
  private:
    // a set of term IDs, one for each "not obseleted" [Term] clause
    std::set<std::string> m_id;

    // a map from alternative ID to its in-use (valid) ID
    std::unordered_map<std::string, std::string> m_valid_id;

    // created in construction by calling _topoorder()
    // which is a topological order of (valid) IDs
    std::vector<std::string> m_ordered_id;

    // term name, one for each valid ID
    std::unordered_map<std::string, std::string> m_name;

    // term depth, one for each valid ID
    std::unordered_map<std::string, index_t> m_depth;

    std::unordered_map<std::string, std::set<std::string>> m_parent;
    std::unordered_map<std::string, std::set<std::string>> m_child;

    index_t m_number_of_terms;

  public:
    Ontology() : m_number_of_terms(0) {}

    // construct a ontology from an obo file;
    Ontology(const std::string&);

    // with extra relationships
    Ontology(const std::string&, const std::set<std::string>&);

    // copy constructor
    Ontology(const Ontology&);

    // assignment constructor
    const Ontology& operator=(const Ontology&);

    bool operator==(const Ontology& other) const;

    std::string get_id(const std::string& id) const { return _validate(id); }

    std::string get_name(const std::string& id) const;

    index_t size() const { return m_number_of_terms; }

    bool empty() const { return m_number_of_terms == 0; }

    std::set<std::string> roots() const;
    double branching_factor() const;
    index_t depth() const; // return the maximum depth of terms in this ontology
    index_t depth(const std::string&) const; // return the depth of a given term

    // relationship queries
    std::set<std::string> parents(const std::string&) const;
    std::set<std::string> children(const std::string&) const;
    std::set<std::string> children(const std::set<std::string>&) const;
    std::set<std::string> ancestors(const std::string&) const;
    std::set<std::string> ancestors(const std::set<std::string>&) const;
    std::set<std::string> offsprings(const std::string&) const;
    std::set<std::string> offsprings(const std::set<std::string>&) const;
    std::set<std::string> leaves() const;

    // Returns leaf terms of the ontology consists of a given list of terms
    std::set<std::string> leaves_of(const std::set<std::string>&) const;

    // Returns the sorted terms vector (root(s)->leaves).
    std::vector<std::string> terms() const { return m_ordered_id; }

    /**
     * Makes a sub-ontology using the given terms
     */
    Ontology subontology(const std::set<std::string>&) const;

    /**
     * Splits the ontology rooted by each root (usually used for Gene ontology)
     */
    std::vector<Ontology> split() const;

    /**
     * @brief Returns the so-called ancestor matrix
     * 
     * @remark corresponding terms are sorted in topological order by terms()
     */
    arma::mat A() const;

    /**
     * @brief Saves one term as (id, name) pair per line to file
     */
    void save_term_as_text(const std::string&) const;

    /**
     * @brief Saves one (parent, child) relation per line to file
     */
    void save_relationship_as_text(const std::string&) const;

    void serialize(std::ostream& stream) const;
    void deserialize(std::istream& stream);

  protected:
    std::string _trim(const std::string&) const;

    // the heavy-lifting actual constructor
    void _load_obo(const std::string& filename,
        const std::set<std::string>& rels = std::set<std::string>());

    // add term with id and name
    bool _add_term(const std::string&, const std::string& = std::string(""));
    void _add_relation(const std::string&, const std::string&);

    /**
     * Updates m_ordered_id which is a topological order of all terms.
     *
     * @remark This function relies on m_parent and m_child and produces m_ordered_id.
     */
    void _topoorder();

    /**
     * Updates m_depth which is the depth of each term.
     *
     * @remark This function relies on the topological order, so it must be called after
     * _topoorder().
     */
    void _update_depth();

    /**
     * Returns the validate term id of a given id.
     *
     * @remark This is useful if an alternative id exists. It relies on the original OBO file that
     * contains [alt_id] fields for some term.
     */
    std::string _validate(const std::string& id) const {
      return CONTAINS(m_valid_id, id)? m_valid_id.at(id) : "";
    }
};

#endif // _LIBCB_ONTOLOGY_H_

// end of file

// -------------
// Yuxiang Jiang (yuxjiang@indiana.edu)
// Department of Computer Science
// Indiana University, Bloomington
// Last modified: Mon 16 Sep 2019 10:25:07 PM P
