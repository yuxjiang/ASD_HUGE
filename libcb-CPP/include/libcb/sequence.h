//! libcb/sequence.h

#ifndef _LIBCB_SEQUENCE_H_
#define _LIBCB_SEQUENCE_H_

#include <istream>
#include <ostream>
#include <string>
#include <cctype>
#include <vector>
#include <utility>
#include <functional>
#include <unordered_map>

#include "util.h"

//! Type of FASTA sequence
struct Sequence {
  std::string id, comment, data;

  Sequence() : id(""), comment(""), data("") {}

  Sequence(
      const std::string& i,
      const std::string& c,
      const std::string& d)
    : id(i), comment(c), data(d) {}

  Sequence(const Sequence& s) {
    id      = s.id;
    comment = s.comment;
    data    = s.data;
  }

  index_t length() const { return data.length(); }

  const Sequence& operator=(const Sequence& s) {
    if (this != &s) {
      id = s.id;
      comment = s.comment;
      data = s.data;
    }
    return *this;
  }

  bool operator<(const Sequence& s) const { return id < s.id; }
  bool operator==(const Sequence& s) const { return id == s.id; }
};

/**
 * @struct Gene
 *
 * @brief a class of entrez gene.
 */
struct Gene : public Sequence {
  /**
   * @brief The default constructor.
   *
   * @param e entrez ID as an `std::string`.
   */
  Gene(const std::string entrez = "n/a") : Sequence() { id = entrez; }

  /**
   * @brief Writes to an output stream.
   *
   * @param out an output stream.
   *
   * @sa deserialize()
   */
  void serialize(std::ostream& out) const {
    uint32_t n = static_cast<uint32_t>(id.length());
    out.write((char*)(&n), sizeof(uint32_t));
    out.write(id.c_str(), n * sizeof(char));
  }

  /**
   * @brief Reads from an input stream.
   *
   * @param in an input stream.
   *
   * @sa sequences()
   */
  void deserialize(std::istream& in) {
    uint32_t n;
    char buf[256]; // buffer for holding entrez ID.
    in.read((char*)(&n), sizeof(uint32_t));
    in.read(buf, n);
    // buf[n] = '\0'; // append a stop char
    // id = buf;
    id = std::string(buf, size_t(n));
  }
};

/**
 * @struct Protein
 *
 * @brief a class for protein sequences.
 */
struct Protein : public Sequence {
  Protein(const std::string& i, const std::string& c, const std::string& d) : Sequence(i, c, d) {
    for (auto & aa : data) {
      aa = toupper(aa);
      if (aa < 'A' || aa > 'Z') {
        warn(__PRETTY_FUNCTION__, "Illegal protein sequence at: " + i + " with " + aa);
      }
    }
  }
};

/**
 * @struct DNA
 *
 * @brief a class for nucleic acid sequences.
 */
struct DNA : public Sequence {
  DNA(const std::string& i, const std::string& c, const std::string& d) : Sequence(i, c, d) {
    for (auto & base : data) {
      base = toupper(base);
      if (base != 'A' && base != 'T' && base != 'C' && base != 'G' && base != 'U') {
        warn(__PRETTY_FUNCTION__, "Illegal nucl sequence at: " + i + " with " + base);
      }
    }
  }
};

/**
 * @brief Reads from a file in FASTA format.
 *
 * @param faa a file in FASTA format
 *
 * @return a vector of sequences of type `Sequence`.
 *
 * @sa fastawrite()
 */
template <typename T>
std::vector<T> fastaread(const std::string& faa) {
  using namespace std;
  ifstream ifs;
  ifs.open(faa, ifstream::in);
  vector<T> seqs;
  string buf;
  string id, comment, data = "";
  while (getline(ifs, buf)) {
    if (buf.empty()) continue;
    if (buf[0] == '>') {
      // encounter a new sequence
      if (!data.empty()) {
        seqs.emplace_back(id, comment, data);
      }
      // update [id], [comment] and [data]
      // >  SEQUENCE_ID  COMMENTS...
      //    ^          ^
      //    pos0       pos1
      size_t pos0 = 1;
      while (pos0 < buf.length() && isspace(buf[pos0])) pos0++;
      size_t pos1 = pos0;
      while (pos1 < buf.length() && !isspace(buf[pos1])) pos1++;
      if (pos1 == buf.length()) {
        id = buf.substr(pos0);
        comment = "";
      } else {
        id = buf.substr(pos0, pos1 - pos0);
        comment = buf.substr(pos1 + 1);
      }
      data = "";
    } else {
      // [data] or a continuation of [data]
      data += trim(buf);
    }
  }
  ifs.close();
  if (!data.empty()) {
    // append the last sequence
    seqs.emplace_back(id, comment, data);
  }
  return seqs;
}

/**
 * @brief Writes to a file in FASTA format.
 *
 * @param faa a file name.
 * @param seqs a vector of sequences of type `Sequence`.
 * @param len the maximum length in each line of the output.
 */
template <typename T>
void fastawrite(const std::string& faa, const std::vector<T>& seqs, index_t len = 80) {
  using namespace std;
  ofstream ofs;
  ofs.open(faa, ofstream::out);
  for (const auto& seq : seqs) {
    ofs << ">" << seq.id;
    if (!seq.comment.empty()) {
      ofs << " " << seq.comment;
    }
    ofs << endl;
    if (len == 0) {
      ofs << seq.data << endl;
    } else {
      string s = seq.data;
      while (s.length() > len) {
        ofs << s.substr(0, len) << endl;
        s = s.substr(len + 1);
      }
      ofs << s << endl;
    }
  }
  ofs.close();
}

namespace std
{
  /**
   * @brief Makes the class `Sequence` hashable
   */
  template<>
    struct hash<Sequence> {
      std::size_t operator()(const Sequence& s) const {
        return hash<std::string>()(s.id);
      }
    };

  template<>
    struct hash<Gene> {
      std::size_t operator()(const Gene& g) const {
        return hash<std::string>()(g.id);
      }
    };

  template<>
    struct hash<Protein> {
      std::size_t operator()(const Protein& p) const {
        return hash<std::string>()(p.id);
      }
    };
}

#endif // _LIBCB_SEQUENCE_H_

// end of file

// -------------
// Yuxiang Jiang (yuxjiang@indiana.edu)
// Department of Computer Science
// Indiana University, Bloomington
// Last modified: Tue 25 Jun 2019 11:28:07 PM P
