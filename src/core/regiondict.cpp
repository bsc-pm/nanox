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

#include "regiondict.hpp"
#include "memorymap.hpp"
#include "atomic.hpp"
#include "version.hpp"
#include "system_decl.hpp"
#include "basethread.hpp"
#include "os.hpp"
#include "globalregt.hpp"

using namespace nanos;

reg_t RegionNode::addNode( nanos_region_dimension_internal_t const *dimensions, unsigned int numDimensions, unsigned int deep, ContainerDense< RegionVectorEntry > &container ) {
   bool lastNode = ( deep == ( 2 * numDimensions - 1 ) );
   std::size_t value = ( ( deep & 1 ) == 0 ) ? dimensions[ (deep >> 1) ].lower_bound : dimensions[ (deep >> 1) ].accessed_length;
   //std::cerr << "this node value is "<< _value << " gonna add value " << value << " this deep " << deep<< std::endl;
   if ( !_sons ) {
      _sons = new std::map<std::size_t, RegionNode *>();
   }

   std::map<std::size_t, RegionNode *>::iterator it = _sons->lower_bound( value );
   bool haveToInsert = ( it == _sons->end() || _sons->key_comp()(value, it->first) );
   reg_t newId = ( lastNode && haveToInsert ) ? container.getNewRegionId() : 0;
   reg_t retId = 0;

   if ( haveToInsert ) {
      it = _sons->insert( it, std::map<std::size_t, RegionNode *>::value_type( value, NEW RegionNode( this, value, newId ) ) );
      if ( lastNode ) container.addRegionNode( it->second );
   }

   if ( lastNode ) {
      retId = it->second->getId();
   } else {
      retId = it->second->addNode( dimensions, numDimensions, deep + 1, container );
   }
   return retId;
}

reg_t RegionNode::checkNode( nanos_region_dimension_internal_t const *dimensions, unsigned int numDimensions, unsigned int deep ) {
   bool lastNode = ( deep == ( 2 * numDimensions - 1 ) );
   std::size_t value = ( ( deep & 1 ) == 0 ) ? dimensions[ (deep >> 1) ].lower_bound : dimensions[ (deep >> 1) ].accessed_length;
   //std::cerr << "this node value is "<< _value << " gonna add value " << value << " this deep " << deep<< std::endl;
   if ( !_sons ) {
      return 0;
   }

   std::map<std::size_t, RegionNode *>::iterator it = _sons->lower_bound( value );
   bool haveToInsert = ( it == _sons->end() || _sons->key_comp()(value, it->first) );
   reg_t retId = 0;

   if ( haveToInsert ) {
      return 0;
   }

   if ( lastNode ) {
      retId = it->second->getId();
   } else {
      retId = it->second->checkNode( dimensions, numDimensions, deep + 1 );
   }
   return retId;
}

namespace nanos {
template <>
void RegionDictionary< ContainerDense >::printRegion( std::ostream &o, reg_t region ) {
    RegionNode const *regNode = this->getRegionNode( region );
    global_reg_t reg( region, this );
    //fprintf(stderr, "%p:%d", this, region);
    o << (void *) this << ":" << std::dec << region;
    if ( regNode == NULL ) {
       //fprintf(stderr, "NULL LEAF !");
       o << "NULL LEAF !";
       return;
    }
    for ( int dimensionCount = this->getNumDimensions() - 1; dimensionCount >= 0; dimensionCount -= 1 ) {  
       std::size_t accessedLength = regNode->getValue();
       regNode = regNode->getParent();
       std::size_t lowerBound = regNode->getValue();
       //fprintf(stderr, "[%zu;%zu]", lowerBound, accessedLength);
       o << "[" << std::dec << lowerBound << ";" << std::dec << accessedLength << "]";
       regNode = regNode->getParent();
    }
    //fprintf(stderr, "{key %p : FAkey %p : Real %p : FAReal %p : %zu : %zu}", (void*)_keyBaseAddress, (void*)reg.getKeyFirstAddress(), (void*)_realBaseAddress, (void*) reg.getRealFirstAddress(), reg.getBreadth(), reg.getDataSize() );
    o << "{key " << (void *) _keyBaseAddress << " : FAkey " << (void*)reg.getKeyFirstAddress() << " : Real " << (void*)_realBaseAddress << " : FAReal " << (void*) reg.getRealFirstAddress() << " : Breadth " << std::dec << reg.getBreadth() << " : DataSize " << std::dec <<  reg.getDataSize() << "}";
}


template <>
void RegionDictionary< ContainerSparse >::printRegion( std::ostream &o, reg_t region ) {
    RegionNode const *regNode = this->getRegionNode( region );
    global_reg_t reg( region, &_orig );
    //fprintf(stderr, "%p:%d", &_orig, region);
    o << "sparse> " << this << ":" << (void *) &_orig << ":" << std::dec << region;
    if ( regNode == NULL ) {
       //fprintf(stderr, "NULL LEAF !");
       o << "NULL LEAD !";
       return;
    }
    for ( int dimensionCount = this->getNumDimensions() - 1; dimensionCount >= 0; dimensionCount -= 1 ) {  
       std::size_t accessedLength = regNode->getValue();
       regNode = regNode->getParent();
       std::size_t lowerBound = regNode->getValue();
       //fprintf(stderr, "[%zu;%zu]", lowerBound, accessedLength);
       o << "[" << std::dec << lowerBound << ";" << std::dec << accessedLength << "]";
       regNode = regNode->getParent();
    }
    //fprintf(stderr, "{key %p : FAkey %p : Real %p : FAReal %p : %zu : %zu}", (void*)_keyBaseAddress, (void*)reg.getKeyFirstAddress(), (void*)_realBaseAddress, (void*) reg.getRealFirstAddress(), reg.getBreadth(), reg.getDataSize() );
    o << "{key " << (void *) _keyBaseAddress << " : FAkey " << (void*)reg.getKeyFirstAddress() << " : Real " << (void*)_realBaseAddress << " : FAReal " << (void*) reg.getRealFirstAddress() << " : Breadth " << std::dec << reg.getBreadth() << " : DataSize " << std::dec << reg.getDataSize() << "}";
}

}

