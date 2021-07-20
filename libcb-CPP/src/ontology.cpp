//!libcb/ontology.cpp

#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <climits>
#include <utility>
#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "../include/libcb/ontology.h"

using namespace std;

Ontology::Ontology(const string& obo_file) {
  _load_obo(obo_file);
}

Ontology::Ontology(const string& obo_file, const set<string>& rels) {
  _load_obo(obo_file, rels);
}

Ontology::Ontology(const Ontology& other) {
  m_id              = other.m_id;
  m_valid_id        = other.m_valid_id;
  m_ordered_id      = other.m_ordered_id;
  m_name            = other.m_name;
  m_depth           = other.m_depth;
  m_parent          = other.m_parent;
  m_child           = other.m_child;
  m_number_of_terms = other.m_number_of_terms;
}

const Ontology& Ontology::operator=(const Ontology& other) {
  if (this != &other) {
    m_id              = other.m_id;
    m_valid_id        = other.m_valid_id;
    m_ordered_id      = other.m_ordered_id;
    m_name            = other.m_name;
    m_depth           = other.m_depth;
    m_parent          = other.m_parent;
    m_child           = other.m_child;
    m_number_of_terms = other.m_number_of_terms;
  }
  return *this;
}

bool Ontology::operator==(const Ontology& other) const {
  if (size() != other.size()) return false;
  for (index_t i = 0; i < m_ordered_id.size(); ++i) {
    if (m_ordered_id[i].compare(other.m_ordered_id[i]) != 0) {
      return false;
    }
  }
  return true;
}

string Ontology::get_name(const string& id) const {
  string vid = _validate(id);
  if (vid.empty()) {
    error_and_exit(__PRETTY_FUNCTION__, string("invalid id: ") + id);
  }
  return m_name.at(vid);
}

set<string> Ontology::roots() const {
  set<string> res;
  for (const auto& parents : m_parent) {
    if (parents.second.empty()) {
      res.insert(parents.first);
    }
  }
  return res;
}

double Ontology::branching_factor() const {
  index_t nonleaf_count(0), child_count(0);
  for (const auto& c : m_child) {
    if (!c.second.empty()) {
      ++ nonleaf_count;
      child_count += c.second.size();
    }
  }
  return static_cast<double>(child_count) / nonleaf_count;
}

index_t Ontology::depth() const {
  index_t max_depth = 0;
  for (const auto& d : m_depth) {
    max_depth = max(max_depth, d.second);
  }
  return max_depth;
}

index_t Ontology::depth(const string& id) const {
  string vid = _validate(id);
  if (vid.empty()) {
    error_and_exit(__PRETTY_FUNCTION__, string("invalid id: ") + id);
  }
  return m_depth.at(vid);
}

set<string> Ontology::parents(const string& id) const {
  string vid = _validate(id);
  if (vid.empty()) {
    error_and_exit(__PRETTY_FUNCTION__, string("invalid id: ") + id);
  }
  return m_parent.at(vid);
}

set<string> Ontology::children(const string& id) const {
  string vid = _validate(id);
  if (vid.empty()) {
    error_and_exit(__PRETTY_FUNCTION__, string("invalid id: ") + id);
  }
  return m_child.at(vid);
}

set<string> Ontology::children(const std::set<string>& ids) const {
  set<string> result;
  for (const auto& id : ids) {
    auto c = children(id);
    result.insert(c.begin(), c.end());
  }
  // self-excluded
  for (const auto& id : ids) {
    result.erase(id);
  }
  return result;
}

set<string> Ontology::ancestors(const string& id) const {
  return ancestors(set<string>({id}));
}

set<string> Ontology::ancestors(const set<string>& ids) const {
  set<string> vids;
  for (const auto& id: ids) {
    string vid = _validate(id);
    if (!vid.empty()) {
      vids.insert(vid);
    }
  }
  set<string> res; // result set
  queue<string> q; // index queue
  for (const auto& id : vids) { q.push(id); }
  while (!q.empty()) {
    auto term = q.front(); q.pop();
    if (CONTAINS(m_id, term)) {
      res.insert(term);
      for (const auto& p : m_parent.at(term)) {
        q.push(p);
      }
    }
  }
  return res;
}

set<string> Ontology::offsprings(const string& id) const {
  set<string> ids; ids.insert(id);
  return offsprings(ids);
}

set<string> Ontology::offsprings(const set<string>& ids) const {
  set<string> vids;
  for (const auto& id: ids) {
    string vid = _validate(id);
    if (!vid.empty()) {
      vids.insert(vid);
    }
  }
  set<string> res; // result set
  queue<string> q; // index queue
  for (const auto& id : vids) { q.push(id); }
  while (!q.empty()) {
    auto term = q.front(); q.pop();
    if (CONTAINS(m_id, term)) {
      res.insert(term);
      for (const auto& p : m_child.at(term)) {
        q.push(p);
      }
    }
  }
  return res;
}

set<string> Ontology::leaves_of(const set<string>& ids) const {
  // create a sub-ontology using these terms
  Ontology ont = subontology(ids);
  return ont.leaves();
}

set<string> Ontology::leaves() const {
  set<string> res;
  for (const auto& id : m_id) {
    if (m_child.at(id).empty()) {
      res.insert(id);
    }
  }
  return res;
}

Ontology Ontology::subontology(const set<string>& ids) const {
  set<string> selected;
  for (const auto& id: ids) {
    string vid = _validate(id);
    if (!vid.empty()) {
      selected.insert(vid);
    }
  }
  Ontology ont(*this);
  for (auto it = m_ordered_id.crbegin(); it != m_ordered_id.crend(); ++it) {
    if (!CONTAINS(selected, (*it))) {
      string vid = *it;
      // disconnect from its children
      for (auto cid : ont.m_child[vid]) { ont.m_parent[cid].erase(vid); }
      // disconnect from its parents
      for (auto pid : ont.m_parent[vid]) { ont.m_child[pid].erase(vid); }
      // let all of its parents adopt all of its children
      for (const auto& pid : ont.m_parent[vid]) {
        for (const auto& cid : ont.m_child[vid]) {
          ont._add_relation(cid, pid);
        }
      }
      ont.m_id.erase(vid);
      ont.m_name.erase(vid);
      ont.m_depth.erase(vid);
      ont.m_parent.erase(vid);
      ont.m_child.erase(vid);
      ont.m_number_of_terms -= 1;
    }
  }
  // update m_valid_id
  ont.m_valid_id.clear();
  for (const auto id: m_valid_id) {
    if (CONTAINS(selected, id.second)) {
      ont.m_valid_id[id.first] = id.second;
    }
  }
  ont._topoorder(); // update ont.m_ordered_id
  ont._update_depth(); // update ont.m_depth
  return ont;
}

vector<Ontology> Ontology::split() const {
  vector<Ontology> res;
  set<string> rts = roots();
  for (const auto& r : rts) {
    res.push_back(subontology(offsprings(r)));
  }
  return res;
}

void Ontology::save_term_as_text(const string& filename) const {
  ofstream ofs;
  ofs.open(filename, ofstream::out);
  for (index_t i = 0; i < m_number_of_terms; ++i) {
    ofs << m_ordered_id[i] << "\t" << m_name.at(m_ordered_id[i]) << endl;
  }
  ofs.close();
}

void Ontology::save_relationship_as_text(const string& filename) const {
  ofstream ofs;
  ofs.open(filename, ofstream::out);
  for (const auto& parent : m_ordered_id) {
    for (const auto& child : m_child.at(parent)) {
      ofs << parent << "\t" << child << endl;
    }
  }
  ofs.close();
}

void Ontology::serialize(ostream& stream) const {
  // variables to store
  // 1. m_ordered_id, don't need to store m_id as it can be rebuilt
  // 2. m_valid_i
  // 3. m_name
  // 4. m_depth
  // 5. m_parent and m_child
  uint32_t data = m_number_of_terms;
  SAVE_VAR(stream, data);
  for (index_t i = 0; i < m_number_of_terms; ++i) {
    string id = m_ordered_id[i];
    // id
    data = id.length();
    SAVE_VAR(stream, data);
    stream.write(id.c_str(), data);
    // name
    data = m_name.at(id).length();
    SAVE_VAR(stream, data);
    stream.write(m_name.at(id).c_str(), data);
    // depth
    data = m_depth.at(id);
    SAVE_VAR(stream, data);
    // parents
    data = m_parent.at(id).size();
    SAVE_VAR(stream, data);
    for (const auto& pid: m_parent.at(id)) {
      uint32_t n = pid.length();
      SAVE_VAR(stream, n);
      stream.write(pid.c_str(), n);
    }
    // children
    data = m_child.at(id).size();
    SAVE_VAR(stream, data);
    for (const auto& cid: m_child.at(id)) {
      uint32_t n = cid.length();
      SAVE_VAR(stream, n);
      stream.write(cid.c_str(), n);
    }
  }
  data = m_valid_id.size();
  SAVE_VAR(stream, data);
  for (const auto& m: m_valid_id) {
    uint32_t n = m.first.length();
    SAVE_VAR(stream, n);
    stream.write(m.first.c_str(), n);
    n = m.second.length();
    SAVE_VAR(stream, n);
    stream.write(m.second.c_str(), n);
  }
}

void Ontology::deserialize(istream& stream) {
  uint32_t data;
  char buf[256];
  LOAD_N(stream, data, sizeof(uint32_t));
  m_number_of_terms = data;
  m_ordered_id.resize(m_number_of_terms);
  m_id.clear();
  m_name.clear();
  m_depth.clear();
  m_parent.clear();
  m_child.clear();
  m_valid_id.clear();
  for (index_t i = 0; i < m_number_of_terms; ++i) {
    // id
    LOAD_N(stream, data, sizeof(uint32_t));
    LOAD_N(stream, buf, data);
    string id(buf, data);
    m_ordered_id[i] = id;
    m_id.insert(id);
    // name
    LOAD_N(stream, data, sizeof(uint32_t));
    LOAD_N(stream, buf, data);
    string name(buf, data);
    m_name[id] = name;
    // depth
    LOAD_N(stream, data, sizeof(uint32_t));
    m_depth[id] = data;
    // parents
    m_parent[id] = set<string>(); // important! insert a record for every id
    LOAD_N(stream, data, sizeof(uint32_t));
    for (uint32_t i = 0; i < data; ++i) {
      uint32_t n;
      LOAD_N(stream, n, sizeof(uint32_t));
      LOAD_N(stream, buf, n);
      m_parent[id].emplace(buf, n);
    }
    // children
    m_child[id] = set<string>();
    LOAD_N(stream, data, sizeof(uint32_t));
    for (uint32_t i = 0; i < data; ++i) {
      uint32_t n;
      LOAD_N(stream, n, sizeof(uint32_t));
      LOAD_N(stream, buf, n);
      m_child[id].emplace(buf, n);
    }
  }
  LOAD_N(stream, data, sizeof(uint32_t));
  for (uint32_t i = 0; i < data; ++i) {
    uint32_t n;
    LOAD_N(stream, n, sizeof(uint32_t));
    LOAD_N(stream, buf, n);
    string src(buf, n);
    LOAD_N(stream, n, sizeof(uint32_t));
    LOAD_N(stream, buf, n);
    string dst(buf, n);
    m_valid_id[src] = dst;
  }
}

void Ontology::_load_obo(const string& obo_file, const set<string>& rels) {
  string buf;
  ifstream ifs;
  ifs.open(obo_file, ifstream::in);

  unordered_map<string, set<string>> alt_ids_list;

  m_number_of_terms = 0;
  while (ifs.good()) {
    getline(ifs, buf);

    if (buf.empty()) continue; // skip empty lines

    if (buf.compare("[Term]") == 0) {
      // encounter a new term
      string this_id, this_name;
      set<string> this_alt_ids, this_is_as;
      unordered_map<string, set<string>> this_rels_of_type;
      bool is_obsolete = false;
      while (ifs.good()) {
        getline(ifs, buf);
        if (buf[0] == '[') {
          // rewind
          int pos = ifs.tellg();
          ifs.seekg(pos - buf.size() - 1, ifs.beg);
          break;
        }
        if (buf.compare(0, 3, "id:") == 0) {
          this_id = _trim(buf);
        }
        if (buf.compare(0, 5, "name:") == 0) {
          this_name = _trim(buf);
        }
        if (buf.compare(0, 7, "alt_id:") == 0) {
          this_alt_ids.insert(_trim(buf));
        }
        if (buf.compare(0, 5, "is_a:") == 0) {
          this_is_as.insert(_trim(buf));
        }
        if (buf.compare(0, 13, "relationship:") == 0) {
          buf = _trim(buf);
          for (const auto& rel : rels) {
            if (buf.compare(0, rel.length(), rel) == 0) {
              buf = ":" + buf.substr(rel.length()); // insert an empty tag to the front
              this_rels_of_type[rel].insert(_trim(buf));
            }
          }
        }
        if (buf.compare(0, 12, "is_obsolete:") == 0) {
          buf = _trim(buf);
          if (buf.compare("true") == 0) {
            is_obsolete = true;
          }
        }
      }

      if (!is_obsolete) {
        // add a new term
        _add_term(this_id, this_name); // similar to set::insert

        // record alt_ids
        alt_ids_list[this_id] = this_alt_ids;

        for (const auto& is_a : this_is_as) {
          _add_relation(this_id, is_a);
        }

        for (const auto& this_rels : this_rels_of_type) {
          for (const auto& this_rel : this_rels.second) {
            _add_relation(this_id, this_rel);
          }
        }
      }
    }
  }

  // add all alt_ids and check on-the-fly
  for (const auto& vid_aids : alt_ids_list) {
    for (const auto& alt_id : vid_aids.second) {
      if (CONTAINS(m_id, alt_id)) {
        error_and_exit(__PRETTY_FUNCTION__,
            string("alt_id: ") + alt_id + string(" of ") + vid_aids.first +
            string(" has an independent entry")
            );
      }
      m_valid_id[alt_id] = vid_aids.first;
    }
  }

#ifdef PFP_DEBUG
  cerr << "[DEBUG] read " << m_number_of_terms << " terms." << endl;
  ifs.close();
#endif

  // update the topological order
  _topoorder();

  // compute depth for each term
  _update_depth();
}

string Ontology::_trim(const string& str) const {
  string res = str;
  if (res.empty()) return res;
  size_t pos, first, last;

  // trim every char after the 1st !
  res = res.substr(0, res.find_first_of('!'));

  // trim every char before the 1st :
  pos = res.find_first_of(':');
  pos = (pos == string::npos ? 0 : pos+1);
  res = res.substr(pos);

  if (res.empty()) return res;
  pos   = res.find_first_not_of(" \t\n\v\f\r");
  first = pos == string::npos? res.length() : pos;
  last  = res.find_last_not_of(" \t\n\v\f\r");
  return res.substr(first, last - first + 1);
}

bool Ontology::_add_term(const string& id, const string& name) {
  pair<set<string>::iterator, bool> ret = m_id.insert(id);

  m_valid_id[id] = id; // valid id is itself
  if (!name.empty()) m_name[id] = name;
  if (ret.second) {
    // new term
    m_parent[id] = set<string>(); // initialize
    m_child[id]  = set<string>();
    ++ m_number_of_terms;
  }
  return ret.second;
}

void Ontology::_add_relation(const string& src, const string& dst) {
  // make sure both src and dst is in the set
  _add_term(src);
  _add_term(dst);

  // dst becomes "a parent" of src
  m_parent[src].insert(dst);
  m_child[dst].insert(src);
}

void Ontology::_topoorder() {
  m_ordered_id.clear();
  queue<string> q;
  unordered_map<string, index_t> holds;
  for (auto& id: m_id) {
    holds[id] = m_parent[id].size();
    if (holds[id] == 0) {
      q.push(id);
    }
  }
  while (!q.empty()) {
    auto term = q.front(); q.pop();
    m_ordered_id.push_back(term);
    for (const auto& child: m_child[term]) {
      holds[child] --;
      if (holds[child] == 0) {
        q.push(child);
      }
    }
  }
  if (m_ordered_id.size() != m_number_of_terms) {
    error_and_exit(__PRETTY_FUNCTION__, "Loop(s) detected!");
  }
}

void Ontology::_update_depth() {
  for (auto it = m_ordered_id.begin(); it != m_ordered_id.end(); ++it) {
    if (m_parent.at(*it).empty()) {
      m_depth[*it] = 0;
    } else {
      index_t d = numeric_limits<index_t>::max();
      for (const auto& p : m_parent.at(*it)) {
        d = min(d, m_depth.at(p));
      }
      m_depth[*it] = d + 1;
    }
  }
}

arma::mat Ontology::A() const {
  arma::mat A(m_number_of_terms, m_number_of_terms, arma::fill::eye);
  auto ordered_terms = m_ordered_id;
  // building an id map from term id back to index
  unordered_map<string, index_t> term_to_index;
  for (index_t i = 0; i < ordered_terms.size(); ++i) {
    term_to_index[ordered_terms[i]] = i;
  }
  // traverse terms in a reversed topological order
  reverse(ordered_terms.begin(), ordered_terms.end());
  for (const auto& term : ordered_terms) {
    index_t j = term_to_index[term];
    for (const auto& parent : parents(term)) {
      index_t i = term_to_index[parent];
      // transfer descendants of [j] to [i]
      A.row(i) = arma::max(A.row(i), A.row(j));
    }
  }
  return A;
}

// -------------
// Yuxiang Jiang (yuxjiang@indiana.edu)
// Department of Computer Science
// Indiana University, Bloomington
// Last modified: Sun 15 Sep 2019 10:07:34 PM P
