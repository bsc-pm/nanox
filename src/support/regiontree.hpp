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

#ifndef _NANOS_REGION_TREE
#define _NANOS_REGION_TREE

#include <algorithm>
//#include <cstddef>
#include <stddef.h>

#include "region.hpp"
#include "regioncollection.hpp"
#include "regionpart.hpp"
#include "regiontreenode.hpp"
#include "regiontree_decl.hpp"


namespace nanos  {
namespace region_tree_private {

//! \struct TraversalNode
//! \brief A structure used to traverse the tree
//! \tparam T type of the data indexed by the region tree
template <typename T>
struct TraversalNode {
#if !REGION_TREE_BOUNDING
   //! Region segment that corresponds to the path already traversed
   Region m_traversedRegionSegment;
#endif
   //! Region segment that corresponds to the path remaining to be traversed up to and including the branch
   Region m_regionSegment;
#if REGION_TREE_BOUNDING
   //! Minimum superset that covers the regions represented by node and its children
   Region m_fullRegion;
#endif
   //! RegionTree node
   typename RegionTree<T>::Node *m_branch;
   
#if REGION_TREE_BOUNDING
   //! \brief Full constructor
   //! \param regionSegment the region segment that corresponds to the path remaining to be traversed up to and including the branch
   //! \param fullRegion minimum superset that covers the regions represented by node and its children
   //! \param branch actual RegionTree node
   TraversalNode(Region const &regionSegment, Region const &fullRegion, typename RegionTree<T>::Node *branch):
      m_regionSegment(regionSegment), m_fullRegion(fullRegion), m_branch(branch)
#else
   //! \brief Full constructor
   //! \param regionSegment the region segment that corresponds to the path remaining to be traversed up to and including the branch
   //! \param branch actual RegionTree node
   TraversalNode(Region const &traversedRegionSegment, Region const &regionSegment, typename RegionTree<T>::Node *branch):
      m_traversedRegionSegment(traversedRegionSegment), m_regionSegment(regionSegment), m_branch(branch)
#endif
      {}
   
   //! \brief Default constructor
#if REGION_TREE_BOUNDING
   TraversalNode(): m_regionSegment(), m_fullRegion(), m_branch(NULL)
#else
   TraversalNode(): m_traversedRegionSegment(), m_regionSegment(), m_branch(NULL)
#endif
      {}
};


//! \struct RegionAndPosition
//! \brief Structure that represents a Region, its position in a list and an invalidation flag.
struct RegionAndPosition {
   //! Indicates if the element is valid.
   bool m_valid;
   //! Region
   Region m_region;
   //! Number of bit splits of the Region segment
   int m_partitionLevel;
   //! Position in the original list
   int m_originalListPosition;
   
   //! \brief Default construct
   RegionAndPosition(): m_valid(false)
      {}
   //! \brief Full constructor
   //! \param region Region
   //! \param partitionLevel number of bit splits of the Region
   //! \param pasition position in the original list
   RegionAndPosition(Region const &region, int partitionLevel, int position):
      m_valid(true), m_region(region), m_partitionLevel(partitionLevel), m_originalListPosition(position)
      {}
};


} // namespace region_tree_private


template<typename T>
inline std::ostream & operator<< (std::ostream &o, RegionTree<T> const &regionTree);
   
using namespace region_tree_private;


/*! \brief Accessor to the nodes in the tree.
    \tparam T type of the data indexed by the region tree
 */
template <typename T>
class RegionTree<T>::iterator {
private:
   //! Pointer to the tree node
   typename RegionTree<T>::Node *m_node;
   //! Region pointed by the node
   Region m_region;
   
public:
   //! \brief Default constructor (the end iterator)
   iterator(): m_node(NULL), m_region()
      {}
   //! \brief Copy constructor
   iterator(typename RegionTree<T>::Node *node, Region const &region): m_node(node), m_region(region)
      {}
   
   //! \brief Checks if it does not point to any node (for instance the end iterator)
   bool isEmpty() const
      { return m_node == NULL; }
   
   //! \brief Do not point to any node
   void clear()
      {
         m_node = NULL;
         m_region = Region();
      }
   
   //! \brief Access the node data
   T const & operator*() const
      { return m_node->m_data; }
   //! \brief Access the node data
   T & operator*()
      { return m_node->m_data; }
   
   //! \brief Get a pointer to the node data
   T const * operator->() const
      { return &m_node->m_data; }
   //! \brief Get a pointer to the node data
   T * operator->()
      { return &m_node->m_data; }
   
   //! \brief Equality comparison
   bool operator==(iterator const &other) const
      { return m_node == other.m_node; }
   
   //! \brief Unequality comparison
   bool operator!=(iterator const &other) const
      { return m_node != other.m_node; }
   
   //! \brief Remove the node from the RegionTree
   void erase()
      {
         typename RegionTree<T>::Node *currentNode = m_node;
         while (currentNode->m_parent != NULL) {
            typename RegionTree<T>::Node *parent = currentNode->m_parent;
            if (parent->m_children[Region::BIT_0] == currentNode) {
               parent->m_children[Region::BIT_0] = NULL;
            } else if (parent->m_children[Region::BIT_1] == currentNode) {
               parent->m_children[Region::BIT_1] = NULL;
            } else {
               parent->m_children[Region::X] = NULL;
            }
            //Tracing::regionTreeRemovedNodes(1);
            delete currentNode;
            
            if (parent->m_children[Region::BIT_0] ==  NULL && parent->m_children[Region::BIT_1] ==  NULL && parent->m_children[Region::X] ==  NULL) {
               // The parent is also empty
               currentNode = parent;
            } else {
#if REGION_TREE_BOUNDING
               parent->recalculateFullRegionFromChildren();
#endif
               // Return since we found a non empty node
               return;
            }
         }
         if (currentNode && currentNode->m_parent == NULL) {
            // We reached the root, and it was empty
            currentNode->clear();
         }
      }
   
   //! \brief Full Region of the node
   Region const & getRegion() const
      { return m_region; }
   
   template<typename T2>
   friend std::ostream &operator<< (std::ostream &o, typename RegionTree<T2>::iterator const &it);
   template<typename T2>
   friend std::ostream &printRecursive (std::ostream &o, typename RegionTree<T2>::iterator const &it);
   friend class RegionTree;
};


template<typename T>
RegionTree<T>::RegionTree(): m_root(0) {
   m_root = NEW Node();
   m_root->init(T());
   //Tracing::regionTreeAddedNodes(1);
}


template<typename T>
void RegionTree<T>::find(iterator_list_t &output, traversal_queue_t &pendingNodes)
{
   while (!pendingNodes.empty()) {
      TraversalNode<T> traversalNode = pendingNodes.back();
      pendingNodes.pop_back();
      typename RegionTree<T>::Node *currentNode = traversalNode.m_branch;
      //Tracing::regionTreeTraversedNodes(1);
      
      Region &regionSegment = traversalNode.m_regionSegment;
      
      if (currentNode->m_regionSegment.containedMatch(regionSegment)) {
#if REGION_TREE_BOUNDING
         Region const &fullRegion = traversalNode.m_fullRegion;
#else
         Region traversedSegment = traversalNode.m_traversedRegionSegment;
         traversedSegment += currentNode->m_regionSegment;
#endif
         regionSegment.advance(currentNode->m_regionSegment.getLength());
         if (regionSegment.getFirstBitNumber() == sizeof(size_t)*8) {
            //Tracing::regionTreeMatchingRegions(1);
#if REGION_TREE_BOUNDING
            output.push_back( iterator(currentNode, fullRegion) );
#else
            output.push_back( iterator(currentNode, traversedSegment) );
#endif
            continue;
         }
         // Match 0
         if (
            regionSegment.firstBitMatches(Region::BIT_0)
            && currentNode->m_children[Region::BIT_0] != NULL
#if REGION_TREE_BOUNDING
            && currentNode->m_children[Region::BIT_0]->m_fullRegion.matches(fullRegion) // Bounding
#endif
         ) {
#if REGION_TREE_BOUNDING
            pendingNodes.push_back( TraversalNode<T>(regionSegment+1, fullRegion, currentNode->m_children[Region::BIT_0]) );
#else
            pendingNodes.push_back( TraversalNode<T>(traversedSegment+Region::BIT_0, regionSegment+1, currentNode->m_children[Region::BIT_0]) );
#endif
         }
         // Match 1
         if (
            regionSegment.firstBitMatches(Region::BIT_1)
            && currentNode->m_children[Region::BIT_1] != NULL
#if REGION_TREE_BOUNDING
            && currentNode->m_children[Region::BIT_1]->m_fullRegion.matches(fullRegion) // Bounding
#endif
         ) {
#if REGION_TREE_BOUNDING
            pendingNodes.push_back( TraversalNode<T>(regionSegment+1, fullRegion, currentNode->m_children[Region::BIT_1]) );
#else
            pendingNodes.push_back( TraversalNode<T>(traversedSegment+Region::BIT_1, regionSegment+1, currentNode->m_children[Region::BIT_1]) );
#endif
         }
         // Match X
         if (
            regionSegment.firstBitMatches(Region::X)
            && currentNode->m_children[Region::X] != NULL
#if REGION_TREE_BOUNDING
            && currentNode->m_children[Region::X]->m_fullRegion.matches(fullRegion) // Bounding
#endif
         ) {
            //Tracing::regionTreeXsInPath(1);
#if REGION_TREE_BOUNDING
            pendingNodes.push_back( TraversalNode<T>(regionSegment+1, fullRegion, currentNode->m_children[Region::X]) );
#else
            pendingNodes.push_back( TraversalNode<T>(traversedSegment+Region::X, regionSegment+1, currentNode->m_children[Region::X]) );
#endif
         }
      }
   }
}


// Returns true if limit reached
template<typename T>
bool RegionTree<T>::findConstrained(iterator_list_t &output, traversal_queue_t &pendingNodes, int limit)
{
   while (!pendingNodes.empty()) {
      TraversalNode<T> traversalNode = pendingNodes.back();
      pendingNodes.pop_back();
      typename RegionTree<T>::Node *currentNode = traversalNode.m_branch;
      //Tracing::regionTreeTraversedNodes(1);
      
      Region &regionSegment = traversalNode.m_regionSegment;
      
      if (currentNode->m_regionSegment.containedMatch(regionSegment)) {
#if REGION_TREE_BOUNDING
         Region const &fullRegion = traversalNode.m_fullRegion;
#else
         Region traversedSegment = traversalNode.m_traversedRegionSegment;
         traversedSegment += currentNode->m_regionSegment;
#endif
         regionSegment.advance(currentNode->m_regionSegment.getLength());
         if (regionSegment.getFirstBitNumber() == sizeof(size_t)*8) {
            //Tracing::regionTreeMatchingRegions(1);
            if (limit == 0) {
               return true;
            }
            limit--;
#if REGION_TREE_BOUNDING
            output.push_back( iterator(currentNode, fullRegion) );
#else
            output.push_back( iterator(currentNode, traversedSegment) );
#endif
            continue;
         }
         // Match 0
         if (
            regionSegment.firstBitMatches(Region::BIT_0)
            && currentNode->m_children[Region::BIT_0] != NULL
#if REGION_TREE_BOUNDING
            && currentNode->m_children[Region::BIT_0]->m_fullRegion.matches(fullRegion) // Bounding
#endif
         ) {
#if REGION_TREE_BOUNDING
            pendingNodes.push_back( TraversalNode<T>(regionSegment+1, fullRegion, currentNode->m_children[Region::BIT_0]) );
#else
            pendingNodes.push_back( TraversalNode<T>(traversedSegment+Region::BIT_0, regionSegment+1, currentNode->m_children[Region::BIT_0]) );
#endif
         }
         // Match 1
         if (
            regionSegment.firstBitMatches(Region::BIT_1)
            && currentNode->m_children[Region::BIT_1] != NULL
#if REGION_TREE_BOUNDING
            && currentNode->m_children[Region::BIT_1]->m_fullRegion.matches(fullRegion) // Bounding
#endif
         ) {
#if REGION_TREE_BOUNDING
            pendingNodes.push_back( TraversalNode<T>(regionSegment+1, fullRegion, currentNode->m_children[Region::BIT_1]) );
#else
            pendingNodes.push_back( TraversalNode<T>(traversedSegment+Region::BIT_1, regionSegment+1, currentNode->m_children[Region::BIT_1]) );
#endif
         }
         // Match X
         if (
            regionSegment.firstBitMatches(Region::X)
            && currentNode->m_children[Region::X] != NULL
#if REGION_TREE_BOUNDING
            && currentNode->m_children[Region::X]->m_fullRegion.matches(fullRegion) // Bounding
#endif
         ) {
            //Tracing::regionTreeXsInPath(1);
#if REGION_TREE_BOUNDING
            pendingNodes.push_back( TraversalNode<T>(regionSegment+1, fullRegion, currentNode->m_children[Region::X]) );
#else
            pendingNodes.push_back( TraversalNode<T>(traversedSegment+Region::X, regionSegment+1, currentNode->m_children[Region::X]) );
#endif
         }
      }
   }
   
   return false;
}


template<typename T>
typename RegionTree<T>::iterator RegionTree<T>::findExactAndMatching(Region const &fullRegion, /* Output */ iterator_list_t &matching, traversal_queue_t &pendingNodes)
{
   iterator exactMatch;
   
   while (!pendingNodes.empty()) {
      TraversalNode<T> traversalNode = pendingNodes.back();
      pendingNodes.pop_back();
      typename RegionTree<T>::Node *currentNode = traversalNode.m_branch;
      //Tracing::regionTreeTraversedNodes(1);
      
      Region &regionSegment = traversalNode.m_regionSegment;
      
      if (currentNode->m_regionSegment.containedMatch(regionSegment)) {
#if !REGION_TREE_BOUNDING
         Region traversedSegment = traversalNode.m_traversedRegionSegment;
         traversedSegment += currentNode->m_regionSegment;
#endif
         regionSegment.advance(currentNode->m_regionSegment.getLength());
         if (regionSegment.getFirstBitNumber() == sizeof(size_t)*8) {
            //Tracing::regionTreeMatchingRegions(1);
#if REGION_TREE_BOUNDING
            if (fullRegion == currentNode->getFullRegion()) {
               exactMatch = iterator(currentNode, fullRegion);
            } else {
               matching.push_back( iterator(currentNode, currentNode->getFullRegion()) );
            }
#else
            if (fullRegion == traversedSegment) {
               exactMatch = iterator(currentNode, traversedSegment);
            } else {
               matching.push_back( iterator(currentNode, traversedSegment) );
            }
#endif
            continue;
         }
         // Match 0
         if (
            regionSegment.firstBitMatches(Region::BIT_0)
            && currentNode->m_children[Region::BIT_0] != NULL
#if REGION_TREE_BOUNDING
            && currentNode->m_children[Region::BIT_0]->m_fullRegion.matches(fullRegion) // Bounding
#endif
         ) {
#if REGION_TREE_BOUNDING
            pendingNodes.push_back( TraversalNode<T>(regionSegment+1, fullRegion, currentNode->m_children[Region::BIT_0]) );
#else
            pendingNodes.push_back( TraversalNode<T>(traversedSegment+Region::BIT_0, regionSegment+1, currentNode->m_children[Region::BIT_0]) );
#endif
         }
         // Match 1
         if (
            regionSegment.firstBitMatches(Region::BIT_1)
            && currentNode->m_children[Region::BIT_1] != NULL
#if REGION_TREE_BOUNDING
            && currentNode->m_children[Region::BIT_1]->m_fullRegion.matches(fullRegion) // Bounding
#endif
         ) {
#if REGION_TREE_BOUNDING
            pendingNodes.push_back( TraversalNode<T>(regionSegment+1, fullRegion, currentNode->m_children[Region::BIT_1]) );
#else
            pendingNodes.push_back( TraversalNode<T>(traversedSegment+Region::BIT_1, regionSegment+1, currentNode->m_children[Region::BIT_1]) );
#endif
         }
         // Match X
         if (
            regionSegment.firstBitMatches(Region::X)
            && currentNode->m_children[Region::X] != NULL
#if REGION_TREE_BOUNDING
            && currentNode->m_children[Region::X]->m_fullRegion.matches(fullRegion) // Bounding
#endif
         ) {
            //Tracing::regionTreeXsInPath(1);
#if REGION_TREE_BOUNDING
            pendingNodes.push_back( TraversalNode<T>(regionSegment+1, fullRegion, currentNode->m_children[Region::X]) );
#else
            pendingNodes.push_back( TraversalNode<T>(traversedSegment+Region::X, regionSegment+1, currentNode->m_children[Region::X]) );
#endif
         }
      }
   }
   
   return exactMatch;
}


template<typename T>
typename RegionTree<T>::iterator RegionTree<T>::findExact(Region const &fullRegion)
{
   Region regionSegment = fullRegion;
   typename RegionTree<T>::Node *currentNode = m_root;
   while (currentNode != NULL) {
      //Tracing::regionTreeTraversedNodes(1);
      
      if (currentNode->m_regionSegment.containedExactMatch(regionSegment)) {
         regionSegment.advance(currentNode->m_regionSegment.getLength());
         if (regionSegment.getFirstBitNumber() == sizeof(size_t)*8) {
            //Tracing::regionTreeMatchingRegions(1);
            return iterator(currentNode, fullRegion);
         }
         // Match 0
         if (
            regionSegment.getFirstBit() == Region::BIT_0
            && currentNode->m_children[Region::BIT_0] != NULL
#if REGION_TREE_BOUNDING
            && currentNode->m_children[Region::BIT_0]->m_fullRegion.matches(fullRegion) // Bounding
#endif
         ) {
            currentNode = currentNode->m_children[Region::BIT_0];
            regionSegment.advance();
         }
         // Match 1
         else if (
            regionSegment.getFirstBit() == Region::BIT_1
            && currentNode->m_children[Region::BIT_1] != NULL
#if REGION_TREE_BOUNDING
            && currentNode->m_children[Region::BIT_1]->m_fullRegion.matches(fullRegion) // Bounding
#endif
         ) {
            currentNode = currentNode->m_children[Region::BIT_1];
            regionSegment.advance();
         }
         // Match X
         else if (
            regionSegment.getFirstBit() == Region::X
            && currentNode->m_children[Region::X] != NULL
#if REGION_TREE_BOUNDING
            && currentNode->m_children[Region::X]->m_fullRegion.matches(fullRegion) // Bounding
#endif
         ) {
            currentNode = currentNode->m_children[Region::X];
            regionSegment.advance();
         }
         else {
            // No match
            return iterator();
         }
      } else {
         // No match
         return iterator();
      }
   }
   
   return iterator();
}


template<typename T>
void RegionTree<T>::addOverlapping(Region const &region, iterator_list_t &output, int partitionLevel)
{
   if ( m_root->m_regionSegment.isEmpty()
        && m_root->m_children[Region::BIT_0] == NULL
        && m_root->m_children[Region::BIT_1] == NULL
        && m_root->m_children[Region::X] == NULL
   ) {
      m_root->m_regionSegment = region;
#if REGION_TREE_BOUNDING
      m_root->m_fullRegion = region;
#endif
      m_root->m_partitionLevel = partitionLevel;
      
      //Tracing::regionTreeMatchingRegions(1);
      ContainerAdapter<iterator_list_t>::insert( output, iterator(m_root, region) );
      return;
   }
   
   typename RegionTree<T>::Node *currentNode = m_root;
   Region regionSegment = region;
   
   while (true) {
      //Tracing::regionTreeTraversedNodes(1);
      
      if (currentNode->m_regionSegment.containedExactMatch(regionSegment)) {
         regionSegment.advance(currentNode->m_regionSegment.getLength());
         if (regionSegment.getFirstBitNumber() == sizeof(size_t)*8) {
            //Tracing::regionTreeMatchingRegions(1);
            currentNode->m_partitionLevel = partitionLevel;
            ContainerAdapter<iterator_list_t>::insert( output, iterator(currentNode, region) );
            return;
         }
         
         // Advance to (and create if necessary) the next node
         Region::bit_value_t childBit = regionSegment.getFirstBit();
         regionSegment.advance();
#if REGION_TREE_BOUNDING
         currentNode->createChildIfNecessary(childBit, region, regionSegment);
#else
         currentNode->createChildIfNecessary(childBit, regionSegment);
#endif
         
#if REGION_TREE_BOUNDING
         // Update node bit string
         currentNode->m_fullRegion |= region;
#endif
         
         currentNode = currentNode->m_children[childBit];
         //Tracing::regionTreeTraversedNodes(1);
      } else {
         // No match possible, so then split the node
#if REGION_TREE_BOUNDING
         typename RegionTree<T>::Node *splittedNode = currentNode->split(regionSegment, region, m_root);
#else
         typename RegionTree<T>::Node *splittedNode = currentNode->split(regionSegment, m_root);
#endif
         //Tracing::regionTreeSplittedNodes(1);
         currentNode = splittedNode;
         
#if REGION_TREE_BOUNDING
         // Update node bit string
         currentNode->m_fullRegion |= region;
#endif
      }
   }
}


template<typename T>
typename RegionTree<T>::iterator RegionTree<T>::addOverlapping(Region const &region)
{
   if ( m_root->m_regionSegment.isEmpty()
        && m_root->m_children[Region::BIT_0] == NULL
        && m_root->m_children[Region::BIT_1] == NULL
        && m_root->m_children[Region::X] == NULL
   ) {
      m_root->m_regionSegment = region;
#if REGION_TREE_BOUNDING
      m_root->m_fullRegion = region;
#endif
      
      //Tracing::regionTreeMatchingRegions(1);
      return iterator(m_root, region);
   }
   
   typename RegionTree<T>::Node *currentNode = m_root;
   Region regionSegment = region;
   
   while (true) {
      //Tracing::regionTreeTraversedNodes(1);
      
      if (currentNode->m_regionSegment.containedExactMatch(regionSegment)) {
         regionSegment.advance(currentNode->m_regionSegment.getLength());
         if (regionSegment.getFirstBitNumber() == sizeof(size_t)*8) {
            //Tracing::regionTreeMatchingRegions(1);
            return iterator(currentNode, region);
         }
         
         // Advance to (and create if necessary) the next node
         Region::bit_value_t childBit = regionSegment.getFirstBit();
         regionSegment.advance();
#if REGION_TREE_BOUNDING
         currentNode->createChildIfNecessary(childBit, region, regionSegment);
#else
         currentNode->createChildIfNecessary(childBit, regionSegment);
#endif
         
#if REGION_TREE_BOUNDING
         // Update node bit string
         currentNode->m_fullRegion |= region;
#endif
         
         currentNode = currentNode->m_children[childBit];
         //Tracing::regionTreeTraversedNodes(1);
      } else {
         // No match possible, so then split the node
#if REGION_TREE_BOUNDING
         typename RegionTree<T>::Node *splittedNode = currentNode->split(regionSegment, region, m_root);
#else
         typename RegionTree<T>::Node *splittedNode = currentNode->split(regionSegment, m_root);
#endif
         //Tracing::regionTreeSplittedNodes(1);
         currentNode = splittedNode;
         
#if REGION_TREE_BOUNDING
         // Update node bit string
         currentNode->m_fullRegion |= region;
#endif
      }
   }
}


template<typename T>
void RegionTree<T>::addOverlappingFrom(Region const &region, Region const &fullRegion, typename RegionTree<T>::Node *from, iterator_list_t &output, int partitionLevel)
{
   if ( m_root->m_regionSegment.isEmpty()
        && m_root->m_children[Region::BIT_0] == NULL
        && m_root->m_children[Region::BIT_1] == NULL
        && m_root->m_children[Region::X] == NULL
   ) {
      m_root->m_regionSegment = fullRegion;
#if REGION_TREE_BOUNDING
      m_root->m_fullRegion = fullRegion;
#endif
      
      //Tracing::regionTreeMatchingRegions(1);
      output.push_back( iterator(m_root, fullRegion) );
      return;
   }
   
   typename RegionTree<T>::Node *currentNode = from;
   Region regionSegment = region;
   
   while (true) {
      //Tracing::regionTreeTraversedNodes(1);
      
      if (currentNode->m_regionSegment.containedExactMatch(regionSegment)) {
         regionSegment.advance(currentNode->m_regionSegment.getLength());
         if (regionSegment.getFirstBitNumber() == sizeof(size_t)*8) {
            //Tracing::regionTreeMatchingRegions(1);
            currentNode->m_partitionLevel = partitionLevel;
            output.push_back( iterator(currentNode, fullRegion) );
            return;
         }
         
         // Advance to (and create if necessary) the next node
         Region::bit_value_t childBit = regionSegment.getFirstBit();
         regionSegment.advance();
#if REGION_TREE_BOUNDING
         currentNode->createChildIfNecessary(childBit, fullRegion, regionSegment);
#else
         currentNode->createChildIfNecessary(childBit, regionSegment);
#endif
         
#if REGION_TREE_BOUNDING
         // Update node bit string
         currentNode->m_fullRegion |= fullRegion;
#endif
         
         currentNode = currentNode->m_children[childBit];
         //Tracing::regionTreeTraversedNodes(1);
      } else {
         // No match possible, so then split the node
#if REGION_TREE_BOUNDING
         typename RegionTree<T>::Node *splittedNode = currentNode->split(regionSegment, fullRegion, m_root);
#else
         typename RegionTree<T>::Node *splittedNode = currentNode->split(regionSegment, m_root);
#endif
         //Tracing::regionTreeSplittedNodes(1);
         currentNode = splittedNode;
         
#if REGION_TREE_BOUNDING
         // Update node bit string
         currentNode->m_fullRegion |= fullRegion;
#endif
      }
   }
}


template<typename T>
template<class output_container_t>
void RegionTree<T>::partition(iterator_list_t &nodes, Region const &region,  bool outputOnlyMatchingPart, bool removeExactMatch, output_container_t &output, int maxPartitioningLevels)
{
   for (typename iterator_list_t::iterator it = nodes.begin(); it != nodes.end(); it++) {
      typename RegionTree<T>::Node *node = it->m_node;
      int originalPartitionLevel = node->m_partitionLevel;
      
      Region nodeRegion = it->getRegion();
      RegionPart nodeRegionSegment(nodeRegion.trim(node->m_regionSegment.getBitsToEnd()), originalPartitionLevel);
      
      Region prefix;
      bool nodeSubsumes = nodeRegionSegment.getPrefixUntilThisSubsumes(region, prefix);
      
#if 0
      {
         typename output_container_t::iterator outputIterator = std::find(output.begin(), output.end(), *it);
         if (outputIterator != output.end()) {
            // The node we are removing has been "added" (although it was already in the tree) to the output, and we are about to remove it from the tree
            output.remove(outputIterator);
         }
      }
#else
      std::remove(output.begin(), output.end(), *it);
#endif
      
      if (nodeSubsumes) {
         T data = node->m_data;
         
         // Find the first ancestor that has at least 2 children and delete everything else
         typename RegionTree<T>::Node *ancestor = NULL;
         
         if (node->m_parent == NULL) {
            node->clear();
         } else {
            typename RegionTree<T>::Node *last = node;
            ancestor = node->m_parent;
            
            while (ancestor != NULL) {
               delete last;
               //Tracing::regionTreeRemovedNodes(1);
               
               if (ancestor->m_children[Region::BIT_0] == last) {
                  if (ancestor->m_children[Region::BIT_1] != NULL || ancestor->m_children[Region::X] != NULL) {
                     ancestor->m_children[Region::BIT_0] = NULL;
#if REGION_TREE_BOUNDING
                     ancestor->recalculateFullRegionFromChildren();
#endif
                     break;
                  }
               } else if (ancestor->m_children[Region::BIT_1] == last) {
                  if (ancestor->m_children[Region::BIT_0] != NULL || ancestor->m_children[Region::X] != NULL) {
                     ancestor->m_children[Region::BIT_1] = NULL;
#if REGION_TREE_BOUNDING
                     ancestor->recalculateFullRegionFromChildren();
#endif
                     break;
                  }
               } else {
                  // X
                  if (ancestor->m_children[Region::BIT_0] != NULL || ancestor->m_children[Region::BIT_1] != NULL) {
                     ancestor->m_children[Region::X] = NULL;
#if REGION_TREE_BOUNDING
                     ancestor->recalculateFullRegionFromChildren();
#endif
                     break;
                  }
               }
               
               last = ancestor;
               ancestor = last->m_parent;
            }
            
            if (ancestor == NULL) {
               // Reached the root node
               last->clear();
            }
         }
         
         typedef std::list<RegionPart> region_part_list_t;
         region_part_list_t parts;
         nodeRegionSegment.partition(region, parts, maxPartitioningLevels);
         
         iterator_list_t newMatchingNodes;
         iterator_list_t newNonMatchingNodes;
         for (typename region_part_list_t::iterator it2 = parts.begin(); it2 != parts.end(); it2++) {
            RegionPart &part = *it2;
            if (removeExactMatch && part == region) {
               continue;
            }
            
            if (ancestor != NULL && nodeRegion.trimFrom(ancestor->m_regionSegment.getFirstBitNumber()).containedExactMatch(part)) {
               // Was not the root node and the previous nodes did not subsume the part (we are not in an X path incorrectly)
               if (part.matches(region)) {
                  addOverlappingFrom(part+ancestor->m_regionSegment.getFirstBitNumber(), part, ancestor, newMatchingNodes, part.getPartitionLevel());
               } else {
                  addOverlappingFrom(part+ancestor->m_regionSegment.getFirstBitNumber(), part, ancestor, newNonMatchingNodes, part.getPartitionLevel());
               }
#if REGION_TREE_BOUNDING
               if (ancestor->m_parent != NULL) {
                  ancestor->m_parent->recalculateFullRegionFromChildren();
               }
#endif
            } else {
               // Either we are in the root node or the node was in a path containing an X that subsumes this part
               if (part.matches(region)) {
                  addOverlapping(part, newMatchingNodes, part.getPartitionLevel());
               } else {
                  addOverlapping(part, newNonMatchingNodes, part.getPartitionLevel());
               }
            }
         }
         // parts.clear();
         
         for (typename iterator_list_t::iterator it2 = newMatchingNodes.begin(); it2 != newMatchingNodes.end(); it2++) {
            iterator &accessor = *it2;
            *accessor = data;
            ContainerAdapter<output_container_t>::insert(output, accessor);
            //Tracing::regionTreeNewFragments(1);
         }
         
         for (typename iterator_list_t::iterator it2 = newNonMatchingNodes.begin(); it2 != newNonMatchingNodes.end(); it2++) {
            iterator &accessor = *it2;
            *accessor = data;
            //Tracing::regionTreeNewFragments(1);
            if (!outputOnlyMatchingPart) {
               ContainerAdapter<output_container_t>::insert(output, accessor);
            }
         }
      } else {
         if (removeExactMatch && nodeRegion == region) {
            // No partition needed. Ignoring exact match
         } else {
            // No partition needed
            ContainerAdapter<output_container_t>::insert(output, *it);
         }
      }
   }
}


template<typename T>
void RegionTree<T>::find(Region const &region, iterator_list_t &output)
{
   traversal_queue_t pendingNodes;
   
#if REGION_TREE_BOUNDING
   pendingNodes.push_back(TraversalNode<T>(region, region, m_root));
#else
   pendingNodes.push_back(TraversalNode<T>(Region(), region, m_root));
#endif
   find(output, pendingNodes);
}


template<typename T>
bool RegionTree<T>::findConstrained(Region const &region, iterator_list_t &output, int limit)
{
   traversal_queue_t pendingNodes;
   
#if REGION_TREE_BOUNDING
   pendingNodes.push_back(TraversalNode<T>(region, region, m_root));
#else
   pendingNodes.push_back(TraversalNode<T>(Region(), region, m_root));
#endif
   
   bool tooMany = findConstrained(output, pendingNodes, limit);
   
   return tooMany;
}


template<typename T>
typename RegionTree<T>::iterator RegionTree<T>::findExactAndMatching(Region const &region, iterator_list_t &matching)
{
   traversal_queue_t pendingNodes;
   
#if REGION_TREE_BOUNDING
   pendingNodes.push_back(TraversalNode<T>(region, region, m_root));
#else
   pendingNodes.push_back(TraversalNode<T>(Region(), region, m_root));
#endif
   return findExactAndMatching(region, matching, pendingNodes);
}


template<typename T>
void RegionTree<T>::insertMissingAndConsolidate(Region const &region, iterator exactMatch, iterator_list_t &matchingParts, iterator_list_t &output, int maxPartitioningLevels)
{
   if (exactMatch.isEmpty() && matchingParts.size() == 0) {
      addOverlapping(region, output, 0);
      return;
   }
   
   // During the part traversal we'll calculate what parts are missing
   RegionCollection<> missingParts;
   missingParts.addPart(region);
   
   std::list<RegionPart> parts;
   
   // Classify the parts according to whether they are full matches or they contain parts outside of the regionSegment
   for (typename iterator_list_t::iterator it = matchingParts.begin(); it != matchingParts.end(); it++) {
      iterator &accessor = *it;
      bool matches, regionSegmentSubsumes, partSubsumes;
      
      region.compare(accessor.getRegion(), /* Outputs: */ matches, regionSegmentSubsumes, partSubsumes);
      if (partSubsumes) {
         // The part has subparts outside of the regionSegment
         if (accessor.m_node->m_partitionLevel < maxPartitioningLevels) {
            // Try to partition it
            RegionPart(accessor.getRegion(), accessor.m_node->m_partitionLevel).partition(region, /* Output */ parts, 0, true, true, maxPartitioningLevels); // Partition into matching, partially matching and non-matching parts
            
            T const data = *accessor;
            remove(accessor);
            
            // Insert the subparts (with correct data)
            iterator_list_t newNode;
            for (std::list<RegionPart>::iterator it2 = parts.begin(); it2 != parts.end(); it2++) {
               RegionPart const &subpart = *it2;
               // We must check if the subpart is already in the parts list
               bool alreadyInThePartsList = false;
               {
                  typename iterator_list_t::iterator it3 = it;
                  it3++;
                  for (; it3 != matchingParts.end(); it3++) {
                     iterator accessor3 = *it3;
                     bool subpartSubsumes, alreadyExistingNodeSubsumes, theyMatch;
                     subpart.compare(accessor3.getRegion(), /* Outputs: */ theyMatch, subpartSubsumes, alreadyExistingNodeSubsumes);
                     if (theyMatch) {
                        if (subpartSubsumes && !alreadyExistingNodeSubsumes) {
                           // The new subpart totally overlaps a part, but we cannot make it any thinner, so we erase the small one
                           matchingParts.remove(it3);
                           accessor3.erase();
                        } else if (alreadyExistingNodeSubsumes && !subpartSubsumes) {
                           // The already existing node totally overlaps the subpart, so we skip it and will decide then if we can make the node thinner
                           alreadyInThePartsList = true;
                        } else {
                           // Either they partially overlap or they do not overlap at all
                        }
                     }
                  }
               }
               
               // A subpart may already exist. Then it must already appear in the list of matching parts. So we skip it.
               if (!alreadyInThePartsList) {
                  addOverlapping(subpart, /* output */ newNode, subpart.getPartitionLevel());
                  iterator newNodeAccessor = *(newNode.begin());
                  newNode.clear();
                  *newNodeAccessor = data;
                  if (subpart.matches(region)) {
                     // A partial or a full match
                     output.push_back(newNodeAccessor);
                     missingParts.substract(subpart, maxPartitioningLevels);
                  }
               }
            }
            
         } else {
            // Partition limiting does not allow us to partition this region
            output.push_back(accessor);
            missingParts.substract(accessor.getRegion(), maxPartitioningLevels);
         }
      } else {
         // The part lies entirely inside the regionSegment
         output.push_back(accessor);
         missingParts.substract(accessor.getRegion(), maxPartitioningLevels);
      }
   }
   
   // If there was no exact match insert any missing parts
   if (exactMatch.isEmpty()) {
      missingParts.defragment();
      for (RegionCollection<>::const_iterator it = missingParts.begin(); it != missingParts.end(); it++) {
         iterator_list_t newNode;
         addOverlapping(*it, /* output */ newNode, 0);
         iterator newNodeAccessor = *(newNode.begin());
         // *newNodeAccessor = data;
         output.push_back(newNodeAccessor);
      }
   }
}


// If it found the whole region (or added it), then it returns it instead of placing it in the outputs
template<typename T>
typename RegionTree<T>::iterator RegionTree<T>::findAndPopulate(Region const &region, iterator_list_t &output, int maxPartitioningLevels)
{
   // Find any matching and exact region
   iterator_list_t matchingNodes;
   iterator wholeRegion = findExactAndMatching(region, matchingNodes);
   
   // Partition the matching nodes according to the region
   partition(matchingNodes, region, /* output only matching */ true, /* remove whole */ !wholeRegion.isEmpty(), output, maxPartitioningLevels);
   
   if (wholeRegion.isEmpty()) {
      // No whole region found, so create the regions that are not present in the tree
      RegionCollection<> alreadyAddedRegionCollection;
      for (typename iterator_list_t::const_iterator it = output.begin(); it != output.end(); it++) {
         RegionPart alreadyAddedRegion(it->getRegion(), it->m_node->m_partitionLevel);
         alreadyAddedRegionCollection.addPart(alreadyAddedRegion);
      }
      
      RegionCollection<> missingRegionCollection( RegionPart(region, 0) );
      missingRegionCollection.substract(alreadyAddedRegionCollection, maxPartitioningLevels);
      
      std::list<RegionPart> const &missingParts = missingRegionCollection.getRegionCollectionReference();
      for (typename std::list<RegionPart>::const_iterator it = missingParts.begin(); it != missingParts.end(); it++) {
         RegionPart const &part = *it;
         //Tracing::regionTreeNewFragments(1);
         // There is no overlap (since these are the missing subregions) so skip the overlap logic
         addOverlapping(part, output, part.getPartitionLevel());
      }
      
      return wholeRegion; // This is empty
   } else {
      // There is a whole region and it overlaps the subregions in the output (if any)
      return wholeRegion;
   }
}


template<typename T>
void RegionTree<T>::findAndPartition(Region const &region, iterator_list_t &output, int maxPartitioningLevels)
{
   iterator_list_t nodes;
   
   find(region, nodes);
   partition(nodes, region, true, output, maxPartitioningLevels);
}


template<typename T>
void RegionTree<T>::remove(iterator const &it)
{
   iterator_list_t removeList;
   
   removeList.insert(it);
   removeMany(removeList);
}


template<typename T>
template<typename ITERATOR_LIST_T>
void RegionTree<T>::removeMany(ITERATOR_LIST_T &removeList)
{
   while (!removeList.empty()) {
      typename RegionTree<T>::Node *node =  ContainerAdapter<ITERATOR_LIST_T>::pop(removeList).m_node;
      //Tracing::regionTreeTraversedNodes(1);
      
      // The root node must not be removed
      if (node->m_parent == NULL) {
         node->clear();
         continue;
      }
      
      typename RegionTree<T>::Node *parent = node->m_parent;
      if (parent->m_children[Region::BIT_0] == node) {
         if (parent->m_children[Region::BIT_1] == NULL && parent->m_children[Region::X] == NULL) {
            ContainerAdapter<ITERATOR_LIST_T>::insert(removeList, iterator(parent, Region()));
         } else {
            parent->m_children[Region::BIT_0] = NULL;
#if REGION_TREE_BOUNDING
            parent->recalculateFullRegionFromChildren();
#endif
         }
      } else if (parent->m_children[Region::BIT_1] == node) {
         if (parent->m_children[Region::BIT_0] == NULL && parent->m_children[Region::X] == NULL) {
            ContainerAdapter<ITERATOR_LIST_T>::insert(removeList, iterator(parent, Region()));
         } else {
            parent->m_children[Region::BIT_1] = NULL;
#if REGION_TREE_BOUNDING
            parent->recalculateFullRegionFromChildren();
#endif
         }
      } else {
         // X
         if (parent->m_children[Region::BIT_0] == NULL && parent->m_children[Region::BIT_1] == NULL) {
            ContainerAdapter<ITERATOR_LIST_T>::insert(removeList, iterator(parent, Region()));
         } else {
            parent->m_children[Region::X] = NULL;
#if REGION_TREE_BOUNDING
            parent->recalculateFullRegionFromChildren();
#endif
         }
      }
      
      delete node;
      //Tracing::regionTreeRemovedNodes(1);
      //Tracing::regionTreeMatchingRegions(1);
   }
}


template<typename T>
void RegionTree<T>::defragment(/* Inout */ iterator_list_t &candidates)
{
   if (candidates.empty()) {
      return;
   }
   
   T data = **(candidates.begin());
   std::list<RegionAndPosition> currentList;
   std::list<RegionAndPosition> nextStepList;
   
   for (unsigned int i=0; i < candidates.size(); i++) {
      currentList.push_back( RegionAndPosition(candidates[i].getRegion(), candidates[i].m_node->m_partitionLevel, i) );
   }
   
   bool effective;
   do {
      effective = false;
      typename std::list<RegionAndPosition>::iterator it = currentList.begin();
      while (it != currentList.end()) {
         Region const &region = it->m_region;
         if (it->m_valid) {
            typename std::list<RegionAndPosition>::iterator it2(it);
            it2++;
            while (it2 != currentList.end()) {
               Region const &region2 = it2->m_region;
               if (it2->m_valid) {
                  Region combinedRegion;
                  //Tracing::regionTreeDefragmentIterations(1);
                  if (region.combine(region2, /* out */ combinedRegion)) {
                     effective = true;
                     
                     if (combinedRegion == region) {
                        it2->m_valid = false;
                     } else if (combinedRegion == region2) {
                        it->m_valid = false;
                        break;
                     } else {
                        int combinedPartLevel;
                        if (it->m_partitionLevel == it2->m_partitionLevel) {
                           if (it->m_partitionLevel > 0) {
                              combinedPartLevel = it->m_partitionLevel - 1;
                           } else {
                              combinedPartLevel = 0;
                           }
                        } else if (it->m_partitionLevel < it2->m_partitionLevel) {
                           combinedPartLevel = it->m_partitionLevel;
                        } else {
                           combinedPartLevel = it2->m_partitionLevel;
                        }
                        
                        it->m_valid = false;
                        it2->m_valid = false;
                        nextStepList.push_back( RegionAndPosition(combinedRegion, combinedPartLevel, -1) );
                        break;
                     }
                     
                  }
               }
               it2++;
            }
            
            if (it->m_valid) {
               nextStepList.push_back(*it);
            }
         }
         it++;
      }
      
      if (effective) {
         currentList.clear();
         currentList.splice(currentList.end(), nextStepList);
      }
      nextStepList.clear();
   } while (effective);
   
   // Find out which nodes should be removed
   bool indexesToBeRemoved[candidates.size()];
   for (unsigned int i = 0; i < candidates.size(); i++) {
      indexesToBeRemoved[i] = true;
   }
   
   for (typename std::list<RegionAndPosition>::iterator it = currentList.begin(); it != currentList.end(); it++) {
      if (it->m_originalListPosition != -1) {
         indexesToBeRemoved[it->m_originalListPosition] = false;
      }
   }
   
   // Separate nodes that must be kept and nodes that must be removed
   iterator_list_t result;
   iterator_list_t toBeRemoved;
   for (unsigned int i = 0; i < candidates.size(); i++) {
      if (indexesToBeRemoved[i]) {
         toBeRemoved.push_back(candidates[i]);
      } else {
         result.push_back(candidates[i]);
      }
   }
   
   remove(toBeRemoved);
   candidates = result;
   
   // Add the new nodes
   for (typename std::list<RegionAndPosition>::iterator it = currentList.begin(); it != currentList.end(); it++) {
      if (it->m_originalListPosition == -1) {
         addOverlapping(it->m_region, /* Output */ candidates, it->m_partitionLevel);
         *(candidates.back()) = data;
      }
   }
}

//! \brief RegionTree iterator formatter
//! \tparam T the type of the contents of the tree
//! \param o the output stream
//! \param it the iterator to be formatted
//! \returns the output stream
template<typename T>
inline std::ostream & operator<< (std::ostream &o, typename RegionTree<T>::iterator const &it)
{
   return o << it.m_node->getFullRegion();
}

//! \brief RegionTree graphviz formatter
//! \tparam T the type of the contents of the tree
//! \param o the output stream
//! \param regionTree the tree to be formatted
//! \returns the output stream
template<typename T>
inline std::ostream & operator<< (std::ostream &o, RegionTree<T> const &regionTree)
{
   o << "digraph {" << std::endl;
   o << "node[shape=record];" << std::endl;
   if (regionTree.m_root != NULL) {
      typename RegionTree<T>::Node const &root = *regionTree.m_root;
      // We would like to do this, but it fails in g++ and xlC
      // o << root;
      // So we do this instead
      printRecursive <T>(o, root);
   }
   return o << "}" << std::endl;;
}

} // namespace nanos

#endif // _NANOS_REGION_TREE
