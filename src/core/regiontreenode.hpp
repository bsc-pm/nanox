/*
   Cell/SMP Superscalar (CellSs/SMPSs): Easy programming the Cell BE/Shared Memory Processors
   Copyright (C) 2008 Barcelona Supercomputing Center - Centro Nacional de Supercomputacion
   
   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.
   
   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
   
   The GNU Lesser General Public License is contained in the file COPYING.
*/

#ifndef _NANOS_REGION_TREE_NODE
#define _NANOS_REGION_TREE_NODE


#include "region.hpp"
#include "regiontreenode_decl.hpp"

#include <ostream>


namespace nanos {


namespace region_tree_private {


template<typename T>
#if REGION_TREE_BOUNDING
RegionTreeNode<T> * RegionTreeNode<T>::split(Region otherRegionSegment, Region const &newFullRegion, RegionTreeNode<T> *& root, bool createNewChild) {
#else
RegionTreeNode<T> * RegionTreeNode<T>::split(Region otherRegionSegment, RegionTreeNode<T> *& root, bool createNewChild) {
#endif
   Region newParentRegion;
   m_regionSegment.getCommonPrefix(otherRegionSegment, newParentRegion);
   
   RegionTreeNode<T> *newParent = new RegionTreeNode<T>();
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
      newParent->m_children[newIndex] = new RegionTreeNode<T>();
      newParent->m_children[newIndex]->init(otherRegionSegment, newParent, T());
#if REGION_TREE_BOUNDING
      newParent->m_children[newIndex]->m_fullRegion = newFullRegion;
#endif
      //Tracing::regionTreeAddedNodes(1);
   }
   
   return newParent;
}


template<typename T>
inline std::ostream &operator<< (std::ostream &o, RegionTreeNode<T> const &regionTreeNode) {
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
inline std::ostream &printRecursive (std::ostream &o, RegionTreeNode<T> const &regionTreeNode) {
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


} // namespace region_tree_private


} // namespace nanos


#endif // _NANOS_REGION_TREE_NODE
