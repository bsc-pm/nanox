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

#ifndef _NANOS_REGION_TREE_NODE_DECL
#define _NANOS_REGION_TREE_NODE_DECL

#include "region.hpp"
#include "regiontree_decl.hpp"


namespace nanos {


/*! \brief Node from the region tree
    \tparam T type of the data indexed by the region tree
*/
template <typename T>
struct RegionTree<T>::Node {
   //! Segment of the Region represented by the node.
   Region m_regionSegment;
#if REGION_TREE_BOUNDING
   //! Minimum superset that covers the regions represented by the children (if not a leaf), or itself (if a leaf).
   Region m_fullRegion;
#endif
   //! Parent node in the tree
   Node *m_parent;
   //! Children nodes indexed by their discriminating in Region::bit_value_t value.
   Node *m_children[3]; // 0, 1, X
   //! Number of bit splits due to partitioning.
   unsigned int m_partitionLevel; // Indicates the number of times that a region has been partitioned until arriving to this node
   //! Actual data indexed by the tree.
   T m_data;
   
   //! \brief Default constructor
   Node(): // This constructor is for the memory manager
#if REGION_TREE_BOUNDING
      m_regionSegment(), m_fullRegion(), m_parent(NULL), m_children(), m_partitionLevel(0), m_data(T())
#else
      m_regionSegment(), m_parent(NULL), m_children(), m_partitionLevel(0), m_data(T())
#endif
      {}
   
   //! \brief Unlinked contructor with data
   //! \param data the data that the node must contain
   Node(T const &data):
#if REGION_TREE_BOUNDING
      m_regionSegment(), m_fullRegion(), m_parent(NULL), m_children(), m_partitionLevel(0), m_data(data)
#else
      m_regionSegment(), m_parent(NULL), m_children(), m_partitionLevel(0), m_data(data)
#endif
      {}
   
   //! \brief Full constructor
   //! \param regionSegment the region segment represented by the node
   //! \param parent the node parent in the tree (it is not linked back)
   //! \param data the data that the node must contain
   Node(Region const &regionSegment, Node *parent, T const &data):
#if REGION_TREE_BOUNDING
      m_regionSegment(regionSegment), m_fullRegion(), m_parent(parent), m_children(), m_partitionLevel(0), m_data(data)
#else
      m_regionSegment(regionSegment), m_parent(parent), m_children(), m_partitionLevel(0), m_data(data)
#endif
      {}
   
   //! \brief Unlinked initializer with data
   //! \param data the data that the node must contain
   void init(T const &data);
   
   //! \brief Full initializer
   //! \param regionSegment the region segment represented by the node
   //! \param parent the node parent in the tree (it is not linked back)
   //! \param data the data that the node must contain
   void init(Region const &regionSegment, Node *parent, T const &data);
   
   //! \brief Clear the node
   void clear();
   
   //! \brief Create a child for a certain Region::bit_value_t if it does not exist yet and return it
   //! \param child bit value of the child
   //! \param regionSegment the region segment of the new child node (if it must be created)
   //! \returns the child
#if REGION_TREE_BOUNDING
   //! \param fullRegion the region of the new child node (if it must be created)
   Node * createChildIfNecessary(Region::bit_value_t child, Region const &fullRegion, Region const &regionSegment);
#else
   Node * createChildIfNecessary(Region::bit_value_t child, Region const &regionSegment);
#endif
   
#if REGION_TREE_BOUNDING
   //! \brief Split the node by a certain region segment
   //! \param otherRegionSegment a region segment with a possible common prefix that determines the actual split point in the region segment.
   //! \param newFullRegion the region that is going to be inserted
   //! \param root a reference to where the root node pointer is stored. May be needed to split the root node.
   //! \param createNewChild determines if a new child must be created that representd the otherRegionSegment
   //! \returns the new parent that covers both region segments
   Node * split(Region otherRegionSegment, Region const &newFullRegion, Node *&root, bool createNewChild = true);
#else
   Node * split(Region otherRegionSegment, Node *&root, bool createNewChild = true);
#endif
   
#if REGION_TREE_BOUNDING
   //! \brief Minimum superset of the regions covered
   //! \returns the minimum superser of the regions covered by the node
   Region const & getFullRegion() const;
   
   //! \brief Set the minimum superset of the regions covered
   //! \param fullRegion the minimum superset of the regions covered
   void setFullRegion(Region const &fullRegion);
   
   //! \brief Update the minimum superset of the regions covered by visiting the children and propagating upwards
   // Recalculation cannot be limited to a number of levels since it may prevent X's from forming upwards, and thus incorrectly bonding searches
   void inline recalculateFullRegionFromChildren();
   
#else
   //! \brief Full Region of the node
   //! \returns the full Region of the node
   Region getFullRegion() const;
#endif
   
};



//! \brief RegionTree node graphviz formatter
//! \tparam T the type of the contents of the tree
//! \param o the output stream
//! \param regionTreeNode the node to be formatted
//! \returns the output stream
template<typename T>
std::ostream &operator<< (std::ostream &o, typename RegionTree<T>::Node const &regionTreeNode);


//! \brief Recursive RegionTree node graphviz formatter 
//! \tparam T the type of the contents of the tree
//! \param o the output stream
//! \param regionTreeNode the node to be formatted
//! \returns the output stream
template<typename T>
std::ostream &printRecursive (std::ostream &o, typename RegionTree<T>::Node const &regionTreeNode);


} // namespace nanos


#endif // _NANOS_REGION_TREE_NODE_DECL
