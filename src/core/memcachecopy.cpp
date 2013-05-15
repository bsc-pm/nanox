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

#if 0
void MemCacheCopy::generateInOps( BaseAddressSpaceInOps &ops, bool input, bool output, WD const &wd ) {
   ops.prepareRegion( _reg, wd );
   if ( input ) {
      unsigned int destination_version = ops.getVersionSetVersion( _reg, output ? ( _version + 1 ) : _version ); //this call is needed for both cases, input and output because it allocates a chunk on the device if needed
      if ( _version != destination_version ) {
         _reg.key->printRegion( _reg.id );
         std::cerr << " version mismatch, transfer needed " << _version << " " << destination_version << std::endl;
         for ( NewLocationInfoList::iterator it = _locations.begin(); it != _locations.end(); it++ ) {
            global_reg_t region_shape( it->first, _reg.key );
            unsigned int version_shape = ops.getVersionNoLock( region_shape );

            if ( _version != version_shape ) {
               global_reg_t data_source( it->second, _reg.key );
               memory_space_id_t location = data_source.getFirstLocation();
               unsigned int version_location = ops.getVersionNoLock( data_source );

               if ( location == 0 ) {
                  std::cerr << "add copy from host, reg " << region_shape.id << " version " << version_location << std::endl;
                  ops.addOpFromHost( region_shape,output ? ( _version + 1 ) : _version /*  _version*/ /* reg */);
               } else {
                  std::cerr << "add copy from device, reg " << region_shape.id << " version " << version_location << std::endl;
                  ops.addOp( &sys.getSeparateMemory( location ) , region_shape, output ? ( _version + 1 ) : _version /*_version*/ /* reg */);
               }
            } else {
               global_reg_t data_source( it->second, _reg.key );
               //memory_space_id_t location = data_source.getFirstLocation();
               unsigned int version_location = ops.getVersionNoLock( data_source );
               std::cerr << "copy not needed: shape version=" << version_shape << " location version= " << version_location << " version wanted " << _version << std::endl;
            }
         }
      }
   } else if ( output ) {
      ops.setRegionVersion( _reg, _version + 1 );
      //only output, no copy is needed, destination memory is set up by "getVersion".
   } else {
      fprintf(stderr, "Error at %s.\n", __FUNCTION__);
   }
}
#endif

void MemCacheCopy::generateInOps2( BaseAddressSpaceInOps &ops, bool input, bool output, WD const &wd ) {
   NANOS_INSTRUMENT( InstrumentState inst3(NANOS_CC_CDIN_GET_ADDR); );
   ops.prepareRegion( _reg, wd );
   NANOS_INSTRUMENT( inst3.close(); );
   NANOS_INSTRUMENT( InstrumentState inst4(NANOS_CC_CDIN_OP_GEN); );
   if ( input ) {
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
