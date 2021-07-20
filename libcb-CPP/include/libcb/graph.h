//! libcb/graph.h

#ifndef _LIBCB_GRAPH_H_
#define _LIBCB_GRAPH_H_

// C library
#include <cmath>
#include <cstdint>
#include <cstring>
// I/O
#include <iostream>
#include <fstream>
#include <sstream>
// container
#include <vector>
// other
#include <unordered_map>
#include <algorithm>
#include <random>
#include <tuple>
#include <utility>
#include <chrono>

#include "util.h"

//! Length of the adjacency matrix of a simple graph
#define LEN_ADJMAT(n) (((n)*(n)-(n))>>1)

/**
 * @brief a class of @a simple @a graph. That is, an undirected graph with
 * neither "self-loops" nor "multi-edges".
 *
 * @pre vertex type must meet the following requirements:
 * @arg A default constructor.
 * @arg A properly defined @a assign operator (i.e., operator=).
 * @arg A pair of member functions (de)serialize to handle binary file I/O.
 *
 * @remark Edges are indicated by a positive float number, any value <= 0
 * indicates the pair of vertices are disconnected.
 */
template<typename T>
class SGraph {
  public:
    //! member types
    using vertex_type        = T;
    using weight_type        = float;
    using edge_type          = std::pair<index_t, index_t>;
    using weighted_edge_type = std::tuple<index_t, index_t, weight_type>;

    //! simple graph merging schemes
    enum merge_scheme {
      maximum,  ///< the maximum of the two edges
      minimum,  ///< the minimum of the two edges
      addition  ///< the sum of the two edges
    };

  private:
    //! A vector of vertices of type T.
    std::vector<vertex_type> m_vertices;

    //! A block of 1D vector to store edge weights.
    /**
     * @remark For the sake of "simple graph", `m_adjmat` does not store
     * self-loop edge weights, thus, it has precisely `n*(n-1)/2` entreis as the
     * size of the upper triangluar matrix, where `n` is the number of vertices.
     */
    weight_type* m_adjmat;

    //! The number of (positive) edges.
    index_t m_number_of_edges;

  public:
    /**
     * @brief The default constructor.
     *
     * @param n the number of vertices.
     */
    SGraph(index_t n = 0) {
      if (n == 0) {
        m_number_of_edges = 0;
        m_adjmat = nullptr;
        return;
      }
      index_t amlen = LEN_ADJMAT(n);
      m_vertices.resize(n);
      m_number_of_edges = 0;
      m_adjmat = new weight_type[amlen];
      memset(m_adjmat, 0, amlen * sizeof(weight_type));
    }

    /**
     * @brief A constructor with vertices only.
     *
     * @param vertices a vector of vertices.
     */
    SGraph(const std::vector<vertex_type>& vertices) {
      m_vertices = vertices;
      index_t n = number_of_vertices();
      index_t amlen = LEN_ADJMAT(n);
      m_adjmat = new weight_type[amlen];
      memset(m_adjmat, 0, amlen * sizeof(weight_type));
      m_number_of_edges = 0;
    }

    /**
     * @brief A constructor with vertices and edges (i.e. V, E).
     *
     * @param vertices a vector of vertices.
     * @param weightedEdges a vector of weighted edges.
     */
    SGraph(
        const std::vector<vertex_type>& vertices,
        const std::vector<weighted_edge_type>& weightedEdges) {
      m_vertices = vertices;
      index_t n = number_of_vertices();
      index_t amlen = LEN_ADJMAT(n);
      m_adjmat = new weight_type[amlen];
      memset(m_adjmat, 0, amlen * sizeof(weight_type));
      m_number_of_edges = 0;
      index_t u, v;
      weight_type w;
      for (const auto& weightedEdge : weightedEdges) {
        std::tie(u, v, w) = weightedEdge;
        set_edge_weight(u, v, w);
      }
    }

    /**
     * @brief A copy constructor.
     *
     * @param g another SGraph<vertex_type> object.
     */
    SGraph(const SGraph& g) {
      index_t n = g.number_of_vertices();
      index_t amlen = LEN_ADJMAT(n);
      m_vertices = g.m_vertices;
      m_number_of_edges = g.m_number_of_edges;
      m_adjmat = new weight_type[amlen];
      memcpy(m_adjmat, g.m_adjmat, amlen * sizeof(weight_type));
    }

    /**
     * @brief The destructor.
     */
    virtual ~SGraph() {
      m_vertices.clear();
      m_number_of_edges = 0;
      if (m_adjmat != nullptr) {
        delete[] m_adjmat;
      }
    }

    /**
     * @brief An assign operator.
     *
     * @param g another SGraph<vertex_type> object.
     *
     * @return A constant reference to itself.
     */
    const SGraph& operator=(const SGraph& g) {
      if (this != &g) {
        index_t n = g.number_of_vertices();
        index_t amlen = LEN_ADJMAT(n);
        if (number_of_vertices() != n) {
          // reallocate memory for [m_adjmat] if necessary
          if (m_adjmat != nullptr) {
            delete[] m_adjmat;
          }
          m_adjmat = new weight_type[amlen];
        }
        m_vertices = g.m_vertices;
        m_number_of_edges = g.m_number_of_edges;
        memcpy(m_adjmat, g.m_adjmat, amlen * sizeof(weight_type));
      }
      return *this;
    }

    /**
     * @brief Resizes the graph (with the number of vertices).
     *
     * @param n the number of vertices.
     */
    void resize(index_t n) {
      index_t amlen = LEN_ADJMAT(n);
      if (number_of_vertices() != n) {
        m_vertices.resize(n);
        if (m_adjmat != nullptr) {
          delete[] m_adjmat;
        }
        m_adjmat = new weight_type[amlen];
      }
      m_number_of_edges = 0;
      memset(m_adjmat, 0, amlen * sizeof(weight_type));
    }

    /**
     * @brief Remove all edges in the networl.
     *
     * @sa remove_edge()
     */
    void remove_all_edges() {
      index_t n = number_of_vertices();
      index_t amlen = LEN_ADJMAT(n);
      m_number_of_edges = 0;
      memset(m_adjmat, 0, amlen * sizeof(weight_type));
    }

    /**
     * @brief Update vertices.
     *
     * @param vertices a vector of vertices.
     */
    void set_vertices(std::vector<vertex_type>& vertices) {
      resize(vertices.size());
      m_vertices = vertices;
    }

    /**
     * @brief Gets a vector of neighbor indices.
     *
     * @param v the index of a vertex.
     *
     * @return A vector of indices.
     */
    std::vector<index_t> neighbor_of(index_t v) const {
      index_t n = number_of_vertices();
      if (v >= n) {
        error_and_exit(__PRETTY_FUNCTION__, "Index out of range.");
      }
      std::vector<index_t> neighbors;
      for (index_t i = 0; i < v; ++i) {
        if (m_adjmat[_edge_index_no_check(i, v)] > 0) {
          neighbors.push_back(i);
        }
      }
      for (index_t i = v + 1; i < n; ++i) {
        if (m_adjmat[_edge_index_no_check(v, i)] > 0) {
          neighbors.push_back(i);
        }
      }
      return neighbors;
    }

    /**
     * @brief Gets the degree of a vertex.
     *
     * @param v the index of a vertex.
     *
     * @return The degree count.
     */
    index_t degree_of(index_t v) const {
      index_t n = number_of_vertices();
      if (v >= n) {
        error_and_exit(__PRETTY_FUNCTION__, "Index out of range.");
      }
      index_t degree = 0;
      for (index_t i = 0; i < v; ++i) {
        if (m_adjmat[_edge_index_no_check(i, v)] > 0) {
          ++degree;
        }
      }
      for (index_t i = v + 1; i < n; ++i) {
        if (m_adjmat[_edge_index_no_check(v, i)] > 0) {
          ++degree;
        }
      }
      return degree;
    }

    /**
     * @brief Gets the number of vertices in the SGraph<vertex_type>.
     *
     * @return The degree count.
     */
    index_t number_of_vertices() const {
      return m_vertices.size();
    }

    /**
     * @brief Gets the number of non-zero weighted edges in the graph.
     *
     * @return The total number of non-zero edges.
     */
    index_t number_of_edges() const {
      return m_number_of_edges;
    }

    /**
     * @brief Gets the number of connected component.
     *
     * @return The number of connected component.
     */
    index_t number_of_connected_component() const {
      index_t n = number_of_vertices();
      std::vector<index_t> parent(n);
      for (index_t i = 0; i < n; ++i) {
        parent[i] = i; // initialization
      }
      std::vector<edge_type> edges = get_edges();
      for (const auto& edge : edges) {
        _uf_union(parent, edge.first, edge.second);
      }
      std::set<index_t> cc_index;
      for (index_t i = 0; i < n; ++i) {
        cc_index.insert(_uf_find_and_flatten(parent, i));
      }
      return static_cast<index_t>(cc_index.size());
    }

    /**
     * @brief Gets the number of singletons.
     *
     * @return The number of singletons.
     */
    index_t number_of_singletons() const {
      index_t n = number_of_vertices();
      index_t res = 0;
      for (index_t i = 0; i < n; ++i) {
        if (degree_of(i) == 0) {
          ++res;
        }
      }
      return res;
    }

    /**
     * @brief Filters edges whose weight is below a cutoff.
     *
     * @param tau a cutoff threshold.
     */
    void filter_edges(weight_type tau) {
      if (tau <= 0.0) return;
      index_t n = number_of_vertices();
      index_t amlen = LEN_ADJMAT(n);
      for (index_t i = 0; i < amlen; ++i) {
        if (m_adjmat[i] > 0.0 && m_adjmat[i] < tau) {
          m_adjmat[i] = 0.0;
          --m_number_of_edges;
        }
      }
    }

    /**
     * @brief Gets a full adjacency matrix by normalizing incoming degrees.
     *
     * @return A full adjacency matrix.
     */
    std::vector<std::vector<float>> in_degree_normalized_adjacency_matrix() const {
      index_t n = number_of_vertices();
      // matrix initialization
      std::vector<std::vector<float>> full_adjmat(n);
      for (auto& row : full_adjmat) {
        row.resize(n);
      }
      std::vector<weight_type> wsum(n, 0.0);
      for (index_t i = 0; i < n; ++i) {
        for (index_t r = 0; r < i; ++r) {
          wsum[i] += m_adjmat[_edge_index_no_check(r, i)];
        }
        for (index_t c = i + 1; c < n; ++c) {
          wsum[i] += m_adjmat[_edge_index_no_check(i, c)];
        }
      }
      for (index_t r = 0; r < n; ++r) {
        for (index_t c = 0; c < n; ++c) {
          if (r == c) {
            full_adjmat[r][c] = 0;
          }
          else {
            full_adjmat[r][c] = static_cast<weight_type>(get_edge_weight(r, c) / wsum[c]);
          }
        }
      }
      return full_adjmat;
    }

    /**
     * @brief Gets a full adjacency matrix by normalizing outgoing degrees.
     *
     * @return A full adjacency matrix.
     */
    std::vector<std::vector<float>> out_degree_normalized_adjacency_matrix() const {
      index_t n = number_of_vertices();
      // matrix initialization
      std::vector<std::vector<float>> full_adjmat(n);
      for (auto& row : full_adjmat) {
        row.resize(n);
      }
      std::vector<weight_type> wsum(n, 0.0);
      for (index_t i = 0; i < n; ++i) {
        for (index_t r = 0; r < i; ++r) {
          wsum[i] += m_adjmat[_edge_index_no_check(r, i)];
        }
        for (index_t c = i + 1; c < n; ++c) {
          wsum[i] += m_adjmat[_edge_index_no_check(i, c)];
        }
      }
      for (index_t r = 0; r < n; ++r) {
        for (index_t c = 0; c < n; ++c) {
          if (r == c) {
            full_adjmat[r][c] = 0;
          }
          else {
            full_adjmat[r][c] = static_cast<weight_type>(get_edge_weight(r, c) / wsum[r]);
          }
        }
      }
      return full_adjmat;
    }

    /**
     * @brief Gets a vector of vertex objects.
     *
     * @return A vector of vertex objects.
     *
     * @sa get_edges()
     */
    std::vector<vertex_type> get_vertices() const {
      return m_vertices;
    }

    /**
     * @brief Gets a vector of edges.
     *
     * @return A vector of edge tuples (src_index, dst_index, weight).
     *
     * @sa get_vertices(), get_edges()
     */
    std::vector<weighted_edge_type> get_weighted_edges() const {
      index_t n = number_of_vertices();
      std::vector<weighted_edge_type> weightedEdges;
      for (index_t r = 0; r < n; ++r) {
        for (index_t c = r + 1; c < n; ++c) {
          weight_type w = m_adjmat[_edge_index_no_check(r, c)];
          if (w > 0) {
            weightedEdges.emplace_back(r, c, w);
          }
        }
      }
      return weightedEdges;
    }

    /**
     * @brief Gets a vector of edges, as index pairs.
     *
     * @return A vector of edge pairs (src_index, dst_index).
     *
     * @sa set_edges()
     */
    std::vector<edge_type> get_edges() const {
      index_t n =  number_of_vertices();
      std::vector<edge_type> edges;
      for (index_t r = 0; r < n; ++r) {
        for (index_t c = r + 1; c < n; ++c) {
          weight_type w = m_adjmat[_edge_index_no_check(r, c)];
          if (w > 0) {
            edges.emplace_back(r, c);
          }
        }
      }
      return edges;
    }

    /**
     * @brief Gets a vector of indices of top [k] hubs.
     *
     * @param k the number of hubs.
     *
     * @return A vector of vertex ID.
     */
    std::vector<index_t> get_hubs(index_t k = 1) const {
      index_t n = number_of_vertices();
      if (k > n) {
        error_and_exit(__PRETTY_FUNCTION__, "K is more than the number of vertices.");
      }
      std::vector<index_t> hubs;
      arma::Col<index_t> degrees(n);
      for (index_t i = 0; i < n; ++i) {
        degrees(i) = degree_of(i);
      }
      arma::uvec indices = arma::sort_index(degrees, "descend");
      // [d]: the degree of k-th hub.
      // Outputs indices of all hubs that have degree at least [d].
      index_t d = degrees(indices(k - 1));
      for (index_t i = 0; i < n; ++i) {
        if (degrees(indices(i)) < d) break;
        hubs.push_back(indices(i));
      }
      return hubs;
    }

    /**
     * @brief Creates a vertex-induced sub-graph.
     *
     * @param indices a vector of vertex indices.
     *
     * @return A SGraph<vertex_type> subgraph.
     */
    SGraph subgraph(const std::vector<index_t>& indices) const {
      // make a vector of vertices
      std::vector<vertex_type> vertices(indices.size());
      for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= number_of_vertices()) {
          error_and_exit(__PRETTY_FUNCTION__, "Index out of range.");
        }
        vertices[i] = m_vertices[indices[i]];
      }
      SGraph g(vertices); // initialize the subgraph.
      index_t n = g.number_of_vertices();
      // copy edge weights
      for (index_t i = 0; i < n; ++i) {
        for (index_t j = i + 1; j < n; ++j) {
          weight_type w = get_edge_weight(indices[i], indices[j]);
          g.set_edge_weight(i, j, w);
        }
      }
      return g;
    }

    /**
     * @brief Creates a vertex-induced graph.
     *
     * @remark Vertices in the new graph do not need to be a subset of the
     * current one.
     *
     * @param vertices a vector of vertices of vertex_type.
     *
     * @return A SGraph<vertex_type> newgraph.
     */
    SGraph project(const std::vector<vertex_type>& vertices) const {
      SGraph g(vertices);
      std::unordered_map<vertex_type, index_t> vertexIndex;
      for (size_t i = 0; i < vertices.size(); ++i) {
        vertexIndex[vertices[i]] = i;
      }
      index_t u, v;
      weight_type w;
      std::vector<vertex_type> verticesSelf = get_vertices();
      std::vector<weighted_edge_type> weightedEdges = get_weighted_edges();
      for (const auto& weightedEdge : weightedEdges) {
        std::tie(u, v, w) = weightedEdge;
        if (CONTAINS(vertexIndex, verticesSelf[u]) && CONTAINS(vertexIndex, verticesSelf[v])) {
          g.set_edge_weight(vertexIndex.at(verticesSelf[u]), vertexIndex.at(verticesSelf[v]), w);
        }
      }
      return g;
    }

    /**
     * @brief Gets edge weight.
     *
     * @param src the source vertex index.
     * @param dst the destination vertex index.
     *
     * @return The weight on edge (src, dst).
     *
     * @sa set_edge_weight().
     */
    weight_type get_edge_weight(index_t src, index_t dst) const {
      index_t ai = _adjmat_index(src, dst);
      index_t n = number_of_vertices();
      return ai >= LEN_ADJMAT(n) ? 0.0 : m_adjmat[ai];
    }

    /**
     * @brief Sets the weight of an edge.
     *
     * @param src the source vertex index.
     * @param dst the destination vertex index.
     * @param w the weight.
     *
     * @sa get_edge_weight().
     */
    void set_edge_weight(
        index_t src,
        index_t dst,
        weight_type w) {
      index_t ai = _adjmat_index(src, dst);
      index_t n = number_of_vertices();
      if (ai >= LEN_ADJMAT(n)) return; // skip invalid edge indices
      _set_edge_weight_with_adjmat_index(ai, w);
    }

    /**
     * @brief Checks if an edge (weight > 0) exists.
     *
     * @param src the source vertex index.
     * @param dst the destination vertex index.
     *
     * @return True or False.
     */
    bool has_edge(index_t src, index_t dst) const {
      index_t ai = _adjmat_index(src, dst);
      index_t n = number_of_vertices();
      if (ai >= LEN_ADJMAT(n)) {
        return false;
      } else {
        return m_adjmat[ai] > 0;
      }
    }

    /**
     * @brief Removes an edge.
     *
     * @param src the @a source vertex index.
     * @param dst the @a destination vertex index.
     *
     * @sa set_edge_weight(), remove_all_edges()
     */
    void remove_edge(index_t src, index_t dst) {
      set_edge_weight(src, dst, 0);
    }

    /**
     * @brief Swaps two edges.
     * In particular, it swaps two edges (a, b), (c, d) to be (a, d), (c, b).
     *
     * @param eab an edge (a, b).
     * @param ecd an edge (c, d).
     */
    void swap_edges(const edge_type& eab,
        const edge_type& ecd) {
      index_t a, b, c, d; // vertex index
      std::tie(a, b) = eab;
      std::tie(c, d) = ecd;
      weight_type wi = get_edge_weight(a, b);
      weight_type wj = get_edge_weight(c, d);
      remove_edge(a, b);
      remove_edge(c, d);
      set_edge_weight(a, d, wi);
      set_edge_weight(c, b, wj);
    }

    /**
     * @brief Disconnets a given vertex from the rest of the graph.
     *
     * @param v the index of a vertex.
     */
    void disconnect(index_t v) {
      std::vector<index_t> neighbors = neighbor_of(v);
      for (const auto& u : neighbors) {
        remove_edge(u, v);
      }
    }

    /**
     * @brief Removes singletons from the graph.
     */
    void remove_singletons() {
      index_t n = number_of_vertices();
      std::vector<index_t> non_singletons;
      for (index_t i = 0; i < n; ++i) {
        if (degree_of(i) > 0) {
          non_singletons.push_back(i);
        }
      }
      (*this) = subgraph(non_singletons);
    }

    /**
     * @brief Merges with another graph.
     *
     * @param other another SGraph object.
     * @param sch a merging scheme.
     *
     * @return A merged SGraph object.
     */
    SGraph merge(const SGraph& other, merge_scheme sch = SGraph::minimum) const {
      // get the union of vertices and their back indices.
      std::vector<index_t> ia, ib;
      std::vector<vertex_type> vertices = vector_union<vertex_type>(m_vertices, other.m_vertices, ia, ib);
      // converts two vector of edge indices, and makes two graphs
      std::vector<weighted_edge_type> wes_self = get_weighted_edges(); // weighted edge from self
      std::vector<weighted_edge_type> wes_other = other.get_weighted_edges(); // weighted edge from other
      index_t u, v;
      weight_type w;
      for (auto& we : wes_self) {
        std::tie(u, v, w) = we;
        we = std::make_tuple(ia[u], ia[v], w); // conversion
      }
      for (auto& we : wes_other) {
        std::tie(u, v, w) = we;
        we = std::make_tuple(ib[u], ib[v], w);
      }
      // merge [other] into [self]
      SGraph g(vertices, wes_self);
      SGraph g_other(vertices, wes_other);
      index_t n = vertices.size();
      index_t amlen = LEN_ADJMAT(n);
      switch(sch) {
        case SGraph::minimum:
          for (index_t i = 0; i < amlen; ++i) {
            g._set_edge_weight_with_adjmat_index(i, std::min(g.m_adjmat[i], g_other.m_adjmat[i]));
          }
          break;
        case SGraph::maximum:
          for (index_t i = 0; i < amlen; ++i) {
            g._set_edge_weight_with_adjmat_index(i, std::max(g.m_adjmat[i], g_other.m_adjmat[i]));
          }
          break;
        case SGraph::addition:
          for (index_t i = 0; i < amlen; ++i) {
            g._set_edge_weight_with_adjmat_index(i, g.m_adjmat[i] + g_other.m_adjmat[i]);
          }
          break;
        default:
          error_and_exit(__PRETTY_FUNCTION__, "Unknown merging scheme.");
      }
      return g;
    }

    /**
     * @brief Writes the object to a binary file.
     *
     * @param out an output stream.
     */
    void serialize(std::ostream& out) const {
      uint32_t n = static_cast<uint32_t>(number_of_vertices());
      out.write((char*)(&n), sizeof(uint32_t));
      for (uint32_t i = 0; i < n; ++i) {
        m_vertices[i].serialize(out);
      }
      uint32_t ne = static_cast<uint32_t>(m_number_of_edges);
      out.write((char*)(&ne), sizeof(uint32_t));
      out.write((char*)m_adjmat, (LEN_ADJMAT(n) * sizeof(weight_type)));
    }

    /**
     * @brief Reads an Graph object from a binary file.
     *
     * @param in an input stream.
     */
    void deserialize(std::istream& in) {
      uint32_t data;
      in.read((char*)(&data), sizeof(uint32_t)); // read the number of vertices
      index_t n = static_cast<index_t>(data);
      resize(n);
      for (index_t i = 0; i < n; ++i) {
        m_vertices[i].deserialize(in); // read each vertex object
      }
      // load weight matrix
      index_t amlen = LEN_ADJMAT(n);
      in.read((char*)(&data), sizeof(uint32_t));
      m_number_of_edges = static_cast<index_t>(data);
      in.read((char*)m_adjmat, amlen * sizeof(weight_type));
    }

    // TODO
    // ----
    //! Shuffles vertices in the simple graph.
    /*!
     * \param pairsToSwap the number of vertex swaps.
     */
    void shuffle_vertices(int pairsToSwap);

    //! Returns the largest connected component.
    SGraph largest_connected_component() const;

    //! Gets the diameter of the graph.
    index_t diameter() const;

  protected:
    /**
     * @brief Converts (src, dst) index pair to internal data block index into
     * `m_adjmat`.
     *
     * @param src the source vertex index.
     * @param dst the destination vertex index.
     *
     * @return The index into `m_adjmat`.
     */
    index_t _adjmat_index(index_t src, index_t dst) const {
      if (src > dst) {
        std::swap(src, dst);
      }
      index_t n = number_of_vertices();
      if (src >= n || dst >= n) {
        return LEN_ADJMAT(n);;
      }
      else if (src == dst) {
        return LEN_ADJMAT(n);;
      }
      return _edge_index_no_check(src, dst);
    }

    /**
     * @brief Converts index without boundary check.
     *
     * @param src the source vertex index.
     * @param dst the destination vertex index.
     *
     * @return The index into `m_adjmat`.
     *
     * @sa _adjmat_index()
     */
    index_t _edge_index_no_check( index_t src, index_t dst) const {
      return (number_of_vertices() * src - (((src + 3) * src) >> 1) + dst - 1);
    }

    /**
     * @brief Sets an edge weight given index into the internal `m_adjmat`.
     *
     * @param ai an index into `m_adjmat`.
     * @param w an edge weight of type `weight_type`.
     */
    void _set_edge_weight_with_adjmat_index(
        index_t ai,
        weight_type w) {
      if (w > 0) {
        if (m_adjmat[ai] <= 0) { // old w <= 0, new w > 0
          ++m_number_of_edges;
        }
      }
      else if (m_adjmat[ai] > 0) { // old w > 0, new w <= 0
        --m_number_of_edges;
      }
      m_adjmat[ai] = w;
    }

    /**
     * @brief unions two groups, part of "union-find" algorithm.
     *
     * @param parent the parent vertex.
     * @param src the source vertex index.
     * @param dst the destination vertex index.
     *
     * @sa _uf_find_and_flattern()
     */
    void _uf_union(
        std::vector<index_t>& parent,
        index_t src,
        index_t dst) const {
      index_t src_root = _uf_find_and_flatten(parent, src);
      index_t dst_root = _uf_find_and_flatten(parent, dst);
      parent[parent[dst_root]] = src_root;
    }

    /**
     * @brief finds the group ancestor, part of "union-find" algorithm.
     *
     * @param parent the parent vertex.
     * @param index the vertex index.
     *
     * @return The ancestor of this group.
     *
     * @sa _uf_union()
     */
    index_t _uf_find_and_flatten(
        std::vector<index_t>& parent,
        index_t index) const {
      index_t i = index;
      while (parent[i] != i) {
        i = parent[i];
      } // find root -> i
      index_t root = i;
      i = index;
      while (parent[i] != i) {
        i = parent[i];
        parent[i] = root;
      }
      return root;
    }
};

/**
 * @brief Randomly rewires an amount of pairs of edges while preserving its
 * degree distribution.
 *
 * @tparam T vertex type.
 * @param g an SGraph<T> object.
 * @param swapTarget the number of pairs of edges to rewire.
 *
 * @return A rewired SGraph<T> object.
 */
template<typename T>
SGraph<T> degree_preserving_edge_rewire(const SGraph<T>& g, int swapTarget) {
  SGraph<T> newGraph = g;
  int edgeCount = newGraph.number_of_edges();
  int scale = 100; // maximum trials = scale * |E|
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::uniform_int_distribution<int> udist(0, edgeCount - 1);
  std::vector<typename SGraph<T>::edge_type> edges = newGraph.get_edges();
  index_t ei, ej; // index of edges
  index_t a, b, c, d; // vertex indices, (a, b) and (c, d)
  int swapCount = 0;
  for (int i = 0; i < scale * edgeCount; ++i) {
    // Draw two edges at random, and try to swap
    ei = udist(generator);
    ej = udist(generator);
    if (ei == ej) continue;
    // Check if the two edges involve 4 distinct vertices.
    // Note that it's guaranteed: a =/= b, c =/= d
    std::tie(a, b) = edges[ei];
    std::tie(c, d) = edges[ej];
    if (a == c || a == d || b == c || b == d) continue;
    // check if the two target edges exist.
    if (newGraph.has_edge(a, d) || newGraph.has_edge(c, b)) continue;
    newGraph.swap_edges(edges[ei], edges[ej]);
    // update edge indicies
    edges[ei] = std::make_pair(a, d); // (a, b) -> (a, d)
    edges[ej] = std::make_pair(c, b); // (c, d) -> (c, b)
    if (++swapCount >= swapTarget) break;
  }
  if (swapCount < swapTarget) {
    warn(__PRETTY_FUNCTION__, "Reached the maximum number of swaps.");
  }
  return newGraph;
}

#endif // _LIBCB_GRAPH_H_

// -------------
// Yuxiang Jiang (yuxjiang@indiana.edu)
// Department of Computer Science
// Indiana University, Bloomington
// Last modified: Tue 25 Jun 2019 11:27:58 PM P
