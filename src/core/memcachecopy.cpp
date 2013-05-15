#include <stdio.h>

#include "memcachecopy_decl.hpp"
#include "system_decl.hpp"
#include "memoryops_decl.hpp"
#include "deviceops.hpp"
#include "workdescriptor.hpp"
MemCacheCopy::MemCacheCopy() : _reg( 0, (reg_key_t) NULL ), _version( 0 ), _locations(), _locationDataReady( false ) {
}

MemCacheCopy::MemCacheCopy( WD const &wd, unsigned int index/*, MemController &ccontrol*/ ) {
   sys.getHostMemory().getRegionId( wd.getCopies()[ index ], _reg );
}

void MemCacheCopy::getVersionInfo() {
   if ( _version == 0 ) {
      sys.getHostMemory().getVersionInfo( _reg, _version, _locations );
      _locationDataReady = true;
   }
}

void MemCacheCopy::generateInOps2( BaseAddressSpaceInOps &ops, bool input, bool output, WD const &wd ) {
   NANOS_INSTRUMENT( InstrumentState inst3(NANOS_CC_CDIN_GET_ADDR); );
   ops.prepareRegion( _reg, wd );
   NANOS_INSTRUMENT( inst3.close(); );
   NANOS_INSTRUMENT( InstrumentState inst4(NANOS_CC_CDIN_OP_GEN); );
   if ( input ) {
      //if ( sys.getNetwork()->getNodeNum() == 0 ) {
      //   fprintf(stderr, "calling copyInputData for wd %d (%p) address %p destination %d\n", wd.getId(), &wd, (void*) _reg.getFirstAddress(), ops.getDestination() );
      //   _reg.printLocationData();
      //}
      ops.copyInputData( _reg, _version, output, _locations );
   } else if ( output ) {
      ops.allocateOutputMemory( _reg, _version + 1 );
   } else {
      fprintf(stderr, "Error at %s.\n", __FUNCTION__);
   }
   NANOS_INSTRUMENT( inst4.close(); );
}

void MemCacheCopy::generateOutOps( SeparateAddressSpaceOutOps &ops, bool input, bool output ) {
   //std::cerr << __FUNCTION__ << std::endl;
}
