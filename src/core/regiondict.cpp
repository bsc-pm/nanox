
#include "regiondict.hpp"
#include "memorymap.hpp"
#include "atomic.hpp"
#include "version.hpp"
#include "system_decl.hpp"
#include "basethread.hpp"
#include "os.hpp"

using namespace nanos;
RegionNode::RegionNode( RegionNode *parent, std::size_t value, reg_t id ) : _parent( parent ), _value( value ), _id( id ), _sons( NULL ) {
   if ( id > 1 ) {
      _memoIntersectInfo = NEW reg_t[ id - 1 ];
      ::memset(_memoIntersectInfo, 0, sizeof( reg_t ) * (id - 1) );
   } else {
      _memoIntersectInfo = NULL;
   }
   //std::cerr << "created node with value " << value << " and id " << id << std::endl;
}

RegionNode::~RegionNode() {
   delete _sons;
}

reg_t RegionNode::getId() const {
   return _id;
}

reg_t RegionNode::getMemoIntersect( reg_t target ) const {
   return _memoIntersectInfo[ target-1 ];
}

void RegionNode::setMemoIntersect( reg_t target, reg_t value ) const {
   _memoIntersectInfo[ target-1 ] = value;
}

reg_t RegionNode::addNode( nanos_region_dimension_internal_t const *dimensions, unsigned int numDimensions, unsigned int deep, ContainerDense< RegionVectorEntry > &container, bool rogue ) {
   bool lastNode = ( deep == ( 2 * numDimensions - 1 ) );
   std::size_t value = ( ( deep & 1 ) == 0 ) ? dimensions[ (deep >> 1) ].lower_bound : dimensions[ (deep >> 1) ].accessed_length;
   //std::cerr << "this node value is "<< _value << " gonna add value " << value << " this deep " << deep<< std::endl;
   if ( !_sons ) {
      _sons = new std::map<std::size_t, RegionNode>();
   }

   std::map<std::size_t, RegionNode>::iterator it = _sons->lower_bound( value );
   bool haveToInsert = ( it == _sons->end() || _sons->key_comp()(value, it->first) );
   reg_t newId = ( lastNode && haveToInsert ) ? container.getNewRegionId() : 0;
   reg_t retId = 0;

   if ( haveToInsert ) {
      it = _sons->insert( it, std::map<std::size_t, RegionNode>::value_type( value, RegionNode( this, value, newId ) ) );
      if ( lastNode ) container.addRegionNode( &(it->second), rogue );
   }

   if ( lastNode ) {
      retId = it->second.getId();
   } else {
      retId = it->second.addNode( dimensions, numDimensions, deep + 1, container, rogue );
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

   std::map<std::size_t, RegionNode>::iterator it = _sons->lower_bound( value );
   bool haveToInsert = ( it == _sons->end() || _sons->key_comp()(value, it->first) );
   reg_t retId = 0;

   if ( haveToInsert ) {
      return 0;
   }

   if ( lastNode ) {
      retId = it->second.getId();
   } else {
      retId = it->second.checkNode( dimensions, numDimensions, deep + 1 );
   }
   return retId;
}

std::size_t RegionNode::getValue() const {
   return _value;
}

RegionNode *RegionNode::getParent() const {
   return _parent;
}

RegionVectorEntry::RegionVectorEntry() : _leaf( NULL ), _data( NULL ) {
}

RegionVectorEntry::RegionVectorEntry( RegionVectorEntry const &rve ) : _leaf( rve._leaf ), _data( rve._data ) {
}

RegionVectorEntry &RegionVectorEntry::operator=( RegionVectorEntry const &rve ) {
   _leaf = rve._leaf;
   _data = rve._data;
   return *this;
}

RegionVectorEntry::~RegionVectorEntry() {
}

void RegionVectorEntry::setLeaf( RegionNode *rn ) {
   _leaf = rn;
}
RegionNode *RegionVectorEntry::getLeaf() const {
   return _leaf;
}

void RegionVectorEntry::setData( Version *d ) {
   _data = d;
}

Version *RegionVectorEntry::getData() const {
   return _data;
}


namespace nanos {
template <>
void RegionDictionary< ContainerDense >::printRegion( reg_t region ) const {
   RegionNode const *regNode = this->getRegionNode( region );
   global_reg_t reg( region, this );
   fprintf(stderr, "%p:%d", this, region);
   if ( regNode == NULL ) {
      fprintf(stderr, "NULL LEAF !");
      return;
   }
   for ( int dimensionCount = this->getNumDimensions() - 1; dimensionCount >= 0; dimensionCount -= 1 ) {  
      std::size_t accessedLength = regNode->getValue();
      regNode = regNode->getParent();
      std::size_t lowerBound = regNode->getValue();
      fprintf(stderr, "[%zu;%zu]", lowerBound, accessedLength);
      regNode = regNode->getParent();
   }
   fprintf(stderr, "{%p:%zu:%zu}", (void*)reg.getFirstAddress(), reg.getBreadth(), reg.getDataSize() );
}

}

