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

#ifndef _NANOS_REGION_TREE_NODE
#define _NANOS_REGION_TREE_NODE


#include "region.hpp"
#include "regiontree_decl.hpp"
#include "regiontreenode_decl.hpp"

#include <ostream>


namespace nanos {


template<typename T>
void RegionTree<T>::Node::init(T const &data) {
   m_regionSegment.clear();
#if REGION_TREE_BOUNDING
   m_fullRegion.clear();
#endif
   m_parent = NULL;
   m_children[Region::BIT_0] = NULL;
   m_children[Region::BIT_1] = NULL;
   m_children[Region::X] = NULL;
   m_partitionLevel = 0;
   m_data = data;
}

template<typename T>
void RegionTree<T>::Node::init(Region const &regionSegment, Node *parent, T const &data) {
   m_regionSegment = regionSegment;
#if REGION_TREE_BOUNDING
   m_fullRegion.clear();
#endif
   m_parent = parent;
   m_children[Region::BIT_0] = NULL;
   m_children[Region::BIT_1] = NULL;
   m_children[Region::X] = NULL;
   m_partitionLevel = 0;
   m_data = data;
}

template<typename T>
void RegionTree<T>::Node::clear() {
   m_regionSegment.clear();
#if REGION_TREE_BOUNDING
   m_fullRegion.clear();
#endif
   m_parent = NULL;
   m_children[Region::BIT_0] = NULL;
   m_children[Region::BIT_1] = NULL;
   m_children[Region::X] = NULL;
   m_partitionLevel = 0;
   m_data = T();
}

template<typename T>
#if REGION_TREE_BOUNDING
typename RegionTree<T>::Node * RegionTree<T>::Node::createChildIfNecessary(Region::bit_value_t child, Region const &fullRegion, Region const &regionSegment) {
#else
typename RegionTree<T>::Node * RegionTree<T>::Node::createChildIfNecessary(Region::bit_value_t child, Region const &regionSegment) {
#endif
   if (m_children[child] == NULL) {
      m_children[child] = NEW Node();
      m_children[child]->init(regionSegment, this, T());
#if REGION_TREE_BOUNDING
      m_children[child]->m_fullRegion = fullRegion;
#endif
      //Tracing::regionTreeAddedNodes(1);
   }
   return m_children[child];
}

#if REGION_TREE_BOUNDING
template<typename T>
Region const & RegionTree<T>::Node::getFullRegion() const {
   return m_fullRegion;
}

template<typename T>
void RegionTree<T>::Node::setFullRegion(Region const &fullRegion) {
   m_fullRegion = fullRegion;
}

template<typename T>
void inline RegionTree<T>::Node::recalculateFullRegionFromChildren() {
   Region oldRegion = m_fullRegion;
   
   if (m_children[Region::BIT_0] !=  NULL) {
      m_fullRegion = m_children[Region::BIT_0]->m_fullRegion;
      if (m_children[Region::BIT_1] !=  NULL) {
         m_fullRegion |= m_children[Region::BIT_1]->m_fullRegion;
      }
      if (m_children[Region::X] !=  NULL) {
         m_fullRegion |= m_children[Region::X]->m_fullRegion;
      }
   } else if (m_children[Region::BIT_1] !=  NULL) {
      m_fullRegion = m_children[Region::BIT_1]->m_fullRegion;
      if (m_children[Region::X] !=  NULL) {
         m_fullRegion |= m_children[Region::X]->m_fullRegion;
      }
   } else if (m_children[Region::X] !=  NULL) {
      m_fullRegion = m_children[Region::X]->m_fullRegion;
   } else {
      m_fullRegion.clear();
   }
   
   if (m_parent !=  NULL && m_fullRegion != oldRegion) {
      //Tracing::regionTreeTraversedNodes(1);
      m_parent->recalculateFullRegionFromChildren();
   }
}

#else
template<typename T>
Region RegionTree<T>::Node::getFullRegion() const {
   Region result;
   if (m_parent != NULL) {
      //Tracing::regionTreeTrackBack(1);
      //Tracing::regionTreeTraversedNodes(1);
      result = m_parent->getFullRegion();
      if (m_parent->m_children[Region::BIT_0] == this) {
         result.addBit(Region::BIT_0);
      } else if (m_parent->m_children[Region::BIT_1] == this) {
         result.addBit(Region::BIT_1);
      } else {
         result.addBit(Region::X);
      }
   }
   result += m_regionSegment;
   return result;
}
#endif


template<typename T>
#if REGION_TREE_BOUNDING
typename RegionTree<T>::Node * RegionTree<T>::Node::split(Region otherRegionSegment, Region const &newFullRegion, Node *& root, bool createNewChild) {
#else
typename RegionTree<T>::Node * RegionTree<T>::Node::split(Region otherRegionSegment, Node *& root, bool createNewChild) {
#endif
   Region newParentRegion;
   m_regionSegment.getCommonPrefix(otherRegionSegment, newParentRegion);
   
   Node *newParent = NEW Node();
   newParent->init(newParentRegion, m_parent, T());
#if REGION_TREE_BOUNDING
   newParent->m_fullRegion = m_fullRegion;
   if (createNewChild) {
      newParent->m_fullRegion.changeBitToX(newParentRegion.getFirstBitNumber() + newParentRegion.getLength());
   }
#endif
   //Tracing::regionTreeAddedNodes(1);
   
   // Relink the parent
   if (m_parent != NULL) {
      if (m_parent->m_children[Region::BIT_0] == this) {
         m_parent->m_children[Region::BIT_0] = newParent;
      } else if (m_parent->m_children[Region::BIT_1] == this) {
         m_parent->m_children[Region::BIT_1] = newParent;
      } else {
         m_parent->m_children[Region::X] = newParent;
      }
   } else {
      root = newParent;
   }
   
   // Link this
   m_regionSegment.advance(newParentRegion.getLength());
   newParent->m_children[m_regionSegment.getFirstBit()] = this;
   m_regionSegment.advance();
   m_parent = newParent;
   
   // Create and link new node
   if (createNewChild) {
      otherRegionSegment.advance(newParentRegion.getLength());
      int newIndex = otherRegionSegment.getFirstBit();
      otherRegionSegment.advance();
      newParent->m_children[newIndex] = new typename RegionTree<T>::Node();
      newParent->m_children[newIndex]->init(otherRegionSegment, newParent, T());
#if REGION_TREE_BOUNDING
      newParent->m_children[newIndex]->m_fullRegion = newFullRegion;
#endif
      //Tracing::regionTreeAddedNodes(1);
   }
   
   return newParent;
}


template<typename T>
inline std::ostream &operator<< (std::ostream &o, typename RegionTree<T>::Node const &regionTreeNode) {
   o << "node" << &regionTreeNode.m_data;
   o << " [shape=record,label=\""
      << "{"
         << "<title> " << &regionTreeNode.m_data
         << "|<title2> " << regionTreeNode.m_regionSegment
         << "|PL: " << regionTreeNode.m_partitionLevel
         << "| " << regionTreeNode.m_data
         << "|{<child0> 0|<child1> 1|<childX> X}"
      << "}"
   << "\"]" << std::endl;
   if (regionTreeNode.m_parent != NULL) {
      if (regionTreeNode.m_parent->m_children[Region::BIT_0] != NULL && &regionTreeNode.m_parent->m_children[Region::BIT_0]->m_data == &regionTreeNode.m_data) {
         o << "node" <<  &regionTreeNode.m_data << ":title" << " -> node" << &regionTreeNode.m_parent->m_data << ":child0" << "[style=dotted]" << std::endl;
      } else if (regionTreeNode.m_parent->m_children[Region::BIT_1] != NULL && &regionTreeNode.m_parent->m_children[Region::BIT_1]->m_data == &regionTreeNode.m_data) {
         o << "node" << &regionTreeNode.m_data << ":title" << " -> node" << &regionTreeNode.m_parent->m_data << ":child1" << "[style=dotted]" << std::endl;
      } else if (regionTreeNode.m_parent->m_children[Region::X] != NULL && &regionTreeNode.m_parent->m_children[Region::X]->m_data == &regionTreeNode.m_data) {
         o << "node" << &regionTreeNode.m_data << ":title" << " -> node" << &regionTreeNode.m_parent->m_data << ":childX" << "[style=dotted]" << std::endl;
      } else {
         o << "LINK_ERROR" << std::endl;
      }
   }
   return o;
}


template<typename T>
inline std::ostream &printRecursive (std::ostream &o, typename RegionTree<T>::Node const &regionTreeNode) {
   o << "node" << &regionTreeNode.m_data;
   o << " [shape=record,label=\""
      << "{"
         << "<title> " << &regionTreeNode.m_data
         << "|<title2> " << regionTreeNode.m_regionSegment
         << "|PL: " << regionTreeNode.m_partitionLevel
         << "| " << regionTreeNode.m_data
         << "|{<child0> 0|<child1> 1|<childX> X}"
      << "}"
   << "\"]" << std::endl;
   if (regionTreeNode.m_parent != NULL) {
      if (regionTreeNode.m_parent->m_children[Region::BIT_0] != NULL && &regionTreeNode.m_parent->m_children[Region::BIT_0]->m_data == &regionTreeNode.m_data) {
         o << "node" <<  &regionTreeNode.m_data << ":title" << " -> node" << &regionTreeNode.m_parent->m_data << ":child0" << "[style=dotted]" << std::endl;
      } else if (regionTreeNode.m_parent->m_children[Region::BIT_1] != NULL && &regionTreeNode.m_parent->m_children[Region::BIT_1]->m_data == &regionTreeNode.m_data) {
         o << "node" << &regionTreeNode.m_data << ":title" << " -> node" << &regionTreeNode.m_parent->m_data << ":child1" << "[style=dotted]" << std::endl;
      } else if (regionTreeNode.m_parent->m_children[Region::X] != NULL && &regionTreeNode.m_parent->m_children[Region::X]->m_data == &regionTreeNode.m_data) {
         o << "node" << &regionTreeNode.m_data << ":title" << " -> node" << &regionTreeNode.m_parent->m_data << ":childX" << "[style=dotted]" << std::endl;
      } else {
         o << "LINK_ERROR" << std::endl;
      }
   }
   if (regionTreeNode.m_children[Region::BIT_0] != NULL) {
      o << "node" <<  &regionTreeNode.m_data << ":child0" << " -> node" << &regionTreeNode.m_children[Region::BIT_0]->m_data << ":title" << std::endl;
      printRecursive <T>(o, *regionTreeNode.m_children[Region::BIT_0]);
   }
   if (regionTreeNode.m_children[Region::BIT_1] != NULL) {
      o << "node" <<  &regionTreeNode.m_data << ":child1" << " -> node" << &regionTreeNode.m_children[Region::BIT_1]->m_data << ":title" << std::endl;
      printRecursive <T>(o, *regionTreeNode.m_children[Region::BIT_1]);
   }
   if (regionTreeNode.m_children[Region::X] != NULL) {
      o << "node" <<  &regionTreeNode.m_data << ":childX" << " -> node" << &regionTreeNode.m_children[Region::X]->m_data << ":title" << std::endl;
      printRecursive <T>(o, *regionTreeNode.m_children[Region::X]);
   }
   return o;
}


} // namespace nanos


#endif // _NANOS_REGION_TREE_NODE
