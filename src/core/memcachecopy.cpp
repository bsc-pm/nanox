#include <stdio.h>

#include "memcachecopy_decl.hpp"
#include "system_decl.hpp"
#include "memoryops_decl.hpp"
#include "deviceops.hpp"
#include "workdescriptor.hpp"

MemCacheCopy::MemCacheCopy() : 
   _version( 0 )
   , _reg( 0, (reg_key_t) NULL )
   , _locations()
   , _locationDataReady( false )
   , _chunk( NULL )
   , _policy( sys.getRegionCachePolicy() )
{
}

MemCacheCopy::MemCacheCopy( WD const &wd, unsigned int index/*, MemController &ccontrol*/ ) {
   sys.getHostMemory().getRegionId( wd.getCopies()[ index ], _reg );
}

void MemCacheCopy::getVersionInfo() {
   if ( _version == 0 ) {
      unsigned int ver = 0;
      sys.getHostMemory().getVersionInfo( _reg, ver, _locations );
      setVersion( ver );
      _locationDataReady = true;
   }
}

void MemCacheCopy::generateInOps( BaseAddressSpaceInOps &ops, bool input, bool output, WD const &wd, unsigned int copyIdx ) {
   //NANOS_INSTRUMENT( InstrumentState inst4(NANOS_CC_CDIN_OP_GEN); );
   if ( input ) {
      ops.copyInputData( *this, wd, copyIdx );
   } else if ( output ) {
      ops.allocateOutputMemory( _reg, _version, wd, copyIdx );
   } else {
      fprintf(stderr, "Error at %s.\n", __FUNCTION__);
   }
   //NANOS_INSTRUMENT( inst4.close(); );
}

void MemCacheCopy::generateOutOps( SeparateMemoryAddressSpace *from, SeparateAddressSpaceOutOps &ops, bool input, bool output, WD const &wd, unsigned int copyIdx ) {
   //std::cerr << __FUNCTION__ << std::endl;
   ops.copyOutputData( from, *this, output, wd, copyIdx );
}

unsigned int MemCacheCopy::getVersion() const {
   return _version;
}

void MemCacheCopy::setVersion( unsigned int version ) {
   _version = version;
}

bool MemCacheCopy::isRooted( memory_space_id_t &loc ) const {
   bool result = _reg.isRooted();
   if ( result ) {
      loc = _reg.getFirstLocation();
   }
   return result;
}

void MemCacheCopy::printLocations( std::ostream &o ) const {
   for ( NewLocationInfoList::const_iterator it = _locations.begin(); it != _locations.end(); it++ ) {
      NewNewDirectoryEntryData *d = NewNewRegionDirectory::getDirectoryEntry( *(_reg.key), it->second );
      o << "   [ " << it->first << "," << it->second << " ] "; _reg.key->printRegion( o, it->first ); 
      if ( d ) o << " " << *d << std::endl; 
      else o << " dir entry n/a" << std::endl;
   }
}
