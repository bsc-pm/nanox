#include <stdint.h>
#include "globalregt_decl.hpp"
#include "newregiondirectory.hpp"
#include "regiondict.hpp"
#include "debug.hpp"

uint64_t global_reg_t::getFirstAddress() const {
   RegionNode *n = key->getRegionNode( id );
   uint64_t baseAddress = key->getBaseAddress();
   uint64_t offset = 0;
   std::vector< std::size_t > const &sizes = key->getDimensionSizes();
   uint64_t acumSizes = 1;

   for ( unsigned int dimIdx = 0; dimIdx < key->getNumDimensions() - 1; dimIdx += 1 ) {
      acumSizes *= sizes[ dimIdx ];
   }
   
   for ( int dimIdx = key->getNumDimensions() - 1; dimIdx >= 0; dimIdx -= 1 ) {
      //std::size_t accessedLength = n->getValue();
      n = n->getParent();
      std::size_t lowerBound = n->getValue();
      n = n->getParent();
      offset += acumSizes * lowerBound;
      if ( dimIdx >= 1 ) acumSizes = acumSizes / sizes[ dimIdx - 1 ];
   }
   return baseAddress + offset; 
}

std::size_t global_reg_t::getBreadth() const {
   RegionNode *n = key->getRegionNode( id );
   std::size_t offset = 0;
   std::size_t lastOffset = 0;
   std::vector< std::size_t > const &sizes = key->getDimensionSizes();
   uint64_t acumSizes = 1;

   for ( unsigned int dimIdx = 0; dimIdx < key->getNumDimensions() - 1; dimIdx += 1 ) {
      acumSizes *= sizes[ dimIdx ];
   }
   
   for ( int dimIdx = key->getNumDimensions() - 1; dimIdx >= 0; dimIdx -= 1 ) {
      std::size_t accessedLength = n->getValue();
      n = n->getParent();
      std::size_t lowerBound = n->getValue();
      n = n->getParent();
      offset += acumSizes * lowerBound;
      lastOffset += acumSizes * ( lowerBound + accessedLength - 1 );
      if ( dimIdx >= 1 ) acumSizes = acumSizes / sizes[ dimIdx - 1 ];
   }
   return ( lastOffset - offset ) + 1;
}

std::size_t global_reg_t::getDataSize() const {
   RegionNode *n = key->getRegionNode( id );
   std::size_t dataSize = 1;

   for ( int dimIdx = key->getNumDimensions() - 1; dimIdx >= 0; dimIdx -= 1 ) {
      std::size_t accessedLength = n->getValue();
      n = n->getParent();
      n = n->getParent();
      dataSize *= accessedLength;
   }
   return dataSize; 
}

unsigned int global_reg_t::getNumDimensions() const {
   return key->getNumDimensions();
}

global_reg_t::global_reg_t( reg_t r, reg_key_t k ) : id( r ), key( k ) {
}

global_reg_t::global_reg_t( reg_t r, const_reg_key_t k ) : id( r ), ckey( k ) {
}

global_reg_t::global_reg_t() : id( 0 ), key( NULL ) {
}

void global_reg_t::fillDimensionData( nanos_region_dimension_internal_t region[]) const {
   RegionNode *n = key->getRegionNode( id );
   std::vector< std::size_t > const &sizes = key->getDimensionSizes();
   for ( int dimIdx = key->getNumDimensions() - 1; dimIdx >= 0; dimIdx -= 1 ) {
      std::size_t accessedLength = n->getValue();
      n = n->getParent();
      std::size_t lowerBound = n->getValue();
      n = n->getParent();
      region[ dimIdx ].accessed_length = accessedLength;
      region[ dimIdx ].lower_bound = lowerBound;
      region[ dimIdx ].size = sizes[ dimIdx ];
   }
}


bool global_reg_t::operator<( global_reg_t const &reg ) const {
   bool result;
   if ( key < reg.key )
      result = true;
   else if ( reg.key < key )
      result = false;
   else result = ( id < reg.id );
   return result;
}

memory_space_id_t global_reg_t::getFirstLocation() const {
   return NewNewRegionDirectory::getFirstLocation( key, id );
}

unsigned int global_reg_t::getHostVersion( bool increaseVersion ) const {
   unsigned int version = 0;
   if ( NewNewRegionDirectory::isLocatedIn( key, id, 0 ) ) {
      version = NewNewRegionDirectory::getVersion( key, id, increaseVersion );
   }
   return version;
}

bool global_reg_t::setCopying( SeparateMemoryAddressSpace &from ) const {
   return true;
}

void global_reg_t::waitCopy( ) const {
}

uint64_t global_reg_t::getBaseAddress() const {
   return key->getBaseAddress();
}

reg_t global_reg_t::getFitRegionId() const {
   RegionNode *n = key->getRegionNode( id );
   bool keep_fitting = true;
   nanos_region_dimension_internal_t fitDimensions[ key->getNumDimensions() ];
   std::vector< std::size_t > const &sizes = key->getDimensionSizes();

   for ( int idx = key->getNumDimensions() - 1; idx >= 0; idx -= 1 ) {
      std::size_t accessedLength = n->getValue();
      n = n->getParent();
      std::size_t lowerBound = n->getValue();
      n = n->getParent();
      fitDimensions[ idx ].size = sizes[ idx ];
      if ( keep_fitting ) {
         fitDimensions[ idx ].accessed_length = accessedLength;
         fitDimensions[ idx ].lower_bound = lowerBound;
         if ( accessedLength != 1 )
            keep_fitting = false;
      } else {
         fitDimensions[ idx ].lower_bound = 0;
         fitDimensions[ idx ].accessed_length = sizes[ idx ];
      }
   }
   return key->obtainRegionId( fitDimensions );
}

void global_reg_t::initializeGlobalEntryIfNeeded() const {
   NewNewDirectoryEntryData *entry = NewNewRegionDirectory::getDirectoryEntry( *key, id );
   if ( entry == NULL ) {
      NewNewRegionDirectory::initializeEntry( key, id );
   }
}

void global_reg_t::setLocationAndVersion( memory_space_id_t loc, unsigned int version ) const {
   NewNewRegionDirectory::addAccess( key, id, loc, version );
}

DeviceOps *global_reg_t::getDeviceOps() const {
   return NewNewRegionDirectory::getOps( key, id );
}

bool global_reg_t::contains( global_reg_t const &reg ) const {
   bool result = false;
   if ( key == reg.key ) {
      if ( key->checkIntersect( id, reg.id ) && ( reg.id == key->computeIntersect( id, reg.id ) ) ) {
         result = true;
      }
   }
   return result;
}

bool global_reg_t::isLocatedIn( memory_space_id_t loc ) const {
   return NewNewRegionDirectory::isLocatedIn( key, id, loc );
}

unsigned int global_reg_t::getVersion() const {
   return NewNewRegionDirectory::getVersion( key, id, false );
}
void global_reg_t::setRooted() const {
   NewNewDirectoryEntryData *entry = NewNewRegionDirectory::getDirectoryEntry( *key, id );
   ensure(entry != NULL, "invalid entry.");
   entry->setRooted();
}

bool global_reg_t::isRooted() const {
   NewNewDirectoryEntryData *entry = NewNewRegionDirectory::getDirectoryEntry( *key, id );
   ensure(entry != NULL, "invalid entry.");
   return entry->isRooted();
}
void global_reg_t::setOwnedMemory(memory_space_id_t loc) const {
   setRooted();
   NewNewRegionDirectory::addRootedAccess( key, id, loc, 1 );
}

std::set< int > const &global_reg_t::getLocations() const {
   NewNewDirectoryEntryData *entry = NewNewRegionDirectory::getDirectoryEntry( *key, id );
   ensure(entry != NULL, "invalid entry.");
   return entry->getLocationsSet();
}

unsigned int global_reg_t::getNumLocations() const {
   NewNewDirectoryEntryData *entry = NewNewRegionDirectory::getDirectoryEntry( *key, id );
   ensure(entry != NULL, "invalid entry.");
   return entry->getLocationsSet().size();
}
