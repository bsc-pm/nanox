/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
/*                                                                                   */
/*      This file is part of the NANOS++ library.                                    */
/*                                                                                   */
/*      NANOS++ is free software: you can redistribute it and/or modify              */
/*      it under the terms of the GNU Lesser General Public License as published by  */
/*      the Free Software Foundation, either version 3 of the License, or            */
/*      (at your option) any later version.                                          */
/*                                                                                   */
/*      NANOS++ is distributed in the hope that it will be useful,                   */
/*      but WITHOUT ANY WARRANTY; without even the implied warranty of               */
/*      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                */
/*      GNU Lesser General Public License for more details.                          */
/*                                                                                   */
/*      You should have received a copy of the GNU Lesser General Public License     */
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/

#ifndef _NANOS_REGION_TREE_DECL
#define _NANOS_REGION_TREE_DECL

//#include <cstddef>
#include <stddef.h>
#include <deque>
#include <list>
#include <ostream>
#include <set>

#include "region_fwd.hpp"
#include "regioncollection_fwd.hpp"
#include "regionpart_fwd.hpp"
#include "regiontree_fwd.hpp"


namespace nanos  {


template<typename T2>
inline std::ostream &operator<< (std::ostream &o, RegionTree<T2> const &regionTree);


namespace region_tree_private {
template <typename T>
struct TraversalNode;
struct RegionAndPosition;
}


using namespace region_tree_private;


/*! \class RegionTree
 *  \brief A data structure that can store data indexed by Region
 *  \tparam T the type of the data indexed by the tree
 */
template<typename T>
class RegionTree {
public:
   // The node structure
   struct Node;
   
   
private:
   //! The root node
   Node *m_root;
   
   
public:
   // An accessor for a leaf node in the tree
   class iterator;
   
   // A list of iterators
   typedef std::list<iterator> iterator_list_t;
   
   
protected:
   // A queue used for traversing the tree
   typedef std::deque< TraversalNode<T> > traversal_queue_t;
   
   
   //! \brief Generate a list of matching nodes according to the list of traversing fronts
   //! \param[out] output the list of matching nodes. Entries are appendend to the origfinal contents
   //! \param[in,out] pendingNodes the list of traversal fronts. Emptied during traversal.
   void find(iterator_list_t &output, traversal_queue_t &pendingNodes);
   
   
   //! \brief Generate a list of matching nodes according to the list of traversing fronts limited to a maximum number of entries
   //! \param[out] output the list of matching nodes. Entries are appendend to the origfinal contents
   //! \param[in,out] pendingNodes the list of traversal fronts. Emptied during traversal.
   //! \param limit maximum number of matches to append
   //! \returns true if th limit has been reached
   bool findConstrained(iterator_list_t &output, traversal_queue_t &pendingNodes, int limit);
   
   //! \brief Generate a list of matching nodes according to the list of traversing fronts and return an exact match if it exists
   //! \param fullRegion the region to look up
   //! \param[out] output the list of matching nodes. Entries are appendend to the origfinal contents
   //! \param[in,out] pendingNodes the list of traversal fronts. Emptied during traversal.
   //! \returns the exact match if it existant, otherwise an empty iterator
   iterator findExactAndMatching(Region const &fullRegion, iterator_list_t &matching, traversal_queue_t &pendingNodes);
   
   
public:
   //! \brief Default constructor
   RegionTree();
   
   //! \brief Find all regions that intersect with a given one
   //! \param region the region
   //! \param[out] output an accessor list to the nodes that intersect
   void find(Region const &region, iterator_list_t &output);
   
   //! \brief Find regions that intersect with a given one up to a given maximum of elements
   //! \param region the region
   //! \param[out] output an accessor list to the nodes that intersect
   //! \param limit the maximum number of entries
   //! \returns true if the limit has been reached
   bool findConstrained(Region const &region, iterator_list_t &output, int limit);
   
   //! \brief Find all regions that intersect with a given one and return the exact match if it exists
   //! \param region the region
   //! \param[out] matching an accessor list to the nodes that intersect
   //! \returns the exact match if it exists, otherwise an empty iterator
   iterator findExactAndMatching(Region const &region, iterator_list_t &matching);
   
   //! \brief Find an exact region
   //! \param region the region
   //! \returns the exact match if it exists, otherwise an empty iterator
   iterator findExact(Region const &region);
   
   //! \brief Find all regions that intersect with a given one, and break their overlapping parts according to a maximum partitioning level
   //! \param region the region
   //! \param[out] output an accessor list to the nodes that intersect
   //! \param maxPartitioningLevels the maximum levels of fragmentation allowed for any region
   void findAndPartition(Region const &region, iterator_list_t &output, int maxPartitioningLevels = -1);
   
   //! \brief Find all regions that intersect with a given one, break their overlapping parts according to a maximum partitioning level, and add any missing subpart to complete it
   //! \param region the region
   //! \param[out] output an accessor list to the nodes that intersect
   //! \param maxPartitioningLevels the maximum levels of fragmentation allowed for any region
   //! \returns the exact match if it exists, otherwise an empty iterator
   iterator findAndPopulate(Region const &region, iterator_list_t &output, int maxPartitioningLevels = -1);
   
   //! \brief Given a region, a list of matching nodes and a possible exact match, partition the overlapping parts according to a maximum partitioning level, and add any missing subpart to complete it
   //! \param region the region
   //! \param exactMatch accessor to the node with the exact match
   //! \param[in,out] accessor input list to the regions in the tree that overlap. Cleared
   //! \param[out] output an accessor list to the new nodes
   //! \param maxPartitioningLevels the maximum levels of fragmentation allowed for any region
   void insertMissingAndConsolidate(Region const &region, iterator exactMatch, iterator_list_t &matchingParts, iterator_list_t &output, int maxPartitioningLevels = -1);
   
   //! \brief Add a region without considering overlaps
   //! \param region the region
   //! \param[out] output an accessor list to the new node
   //! \param partitionLevel partition level of the new node
   void addOverlapping(Region const &region, iterator_list_t &output, int partitionLevel=0);
   
   //! \brief Add a region without considering overlaps
   //! \param region the region
   //! \returns an accessor to the new node
   iterator addOverlapping(Region const &region);
   
   //! \brief Add a region without considering overlaps below a specific point in the tree
   //! \param region the region segment
   //! \param fullRegion the region
   //! \param from the starting point in the tree
   //! \param[out] output an accessor list to the new node
   //! \param partitionLevel partition level of the new node
   void addOverlappingFrom(Region const &region, Region const &fullRegion, Node *from, iterator_list_t &output, int partitionLevel=0);
   
   //! \brief Remove an element from the tree
   //! \param it an iterator pointing to the element
   void remove(iterator const &it);
   
   //! \brief Remove a list of elements from the tree
   //! \tparam ITERATOR_LIST_T the type of the list
   //! \param removeList the list of iterators that point to the elements to be removed
   template<typename ITERATOR_LIST_T>
   void removeMany(ITERATOR_LIST_T &removeList);
   
   //! \brief Coalesce a list of fragments in the tree into a smaller one preserving their semantics
   //! \param[in,out] cadidates on input a list of iterators to the candidate nodes, and on output a list of the final nodes
   void defragment(/* Inout */ iterator_list_t &candidates);
   
   //! \brief Partition a set of nodes according to their intersection with another region
   //! \tparam output_container_t the type of the output container. Usually std::set<iterator>
   //! \param nodes list of iterators corresponding to the (leaf) nodes to be partitioned
   //! \param region the region that determines the partitioning boundary
   //! \param outputOnlyMatchingPart true if the output should receive only the list of intersecting subregions and false if it should receive all fragments
   //! \param removeExactMatch true if a node containing the exact region should be removed
   //! \param[out] output the list of (intersecting if \a outputOnlyMatchingPart is true) fragments
   //! \param  maxPartitioningLevels the maximum levels of fragmentation allowed for any region
   template <class output_container_t>
   void partition(iterator_list_t &nodes, Region const &region, bool outputOnlyMatchingPart, bool removeExactMatch, output_container_t &output, int maxPartitioningLevels = -1);
   
   //! \brief RegionTree stream formatter in dot format
   //! \tparam T2 the type of the contents of the tree
   //! \param o the output stream
   //! \param regionTree the region tree
   //! \returns the output stream
   template<typename T2>
   friend std::ostream & operator<< (std::ostream &o, RegionTree<T2> const &regionTree);
   
};


} // namespace nanos


#endif // _NANOS_REGION_TREE_DECL
