#include "memcontroller_decl.hpp"
#include "workdescriptor.hpp"
#include "regiondict.hpp"
#include "newregiondirectory.hpp"
namespace nanos {
MemController::MemController( WD const &wd ) : _wd( wd ), _memorySpaceId( ( memory_space_id_t ) -1 ), _provideLock(), _providedRegions() {
   if ( _wd.getNumCopies() > 0 ) {
      _memCacheCopies = NEW MemCacheCopy[ wd.getNumCopies() ];
   }
}

bool MemController::hasVersionInfoForRegion( global_reg_t reg, unsigned int &version, NewLocationInfoList &locations ) {
   bool resultHIT = false;
   bool resultSUBR = false;
   bool resultSUPER = false;
   std::map<NewNewRegionDirectory::RegionDirectoryKey, std::map< reg_t, unsigned int > >::iterator wantedDir = _providedRegions.find( reg.key );
   if ( wantedDir != _providedRegions.end() ) {
      unsigned int versionHIT = 0;
      std::map< reg_t, unsigned int >::iterator wantedReg = wantedDir->second.find( reg.id );
      if ( wantedReg != wantedDir->second.end() ) {
         versionHIT = wantedReg->second;
         resultHIT = true;
         wantedDir->second.erase( wantedReg );
      }

      unsigned int versionSUPER = 0;
      reg_t superPart = wantedDir->first->isThisPartOf( reg.id, wantedDir->second.begin(), wantedDir->second.end(), versionSUPER ); 
      if ( superPart != 0 ) {
         resultSUPER = true;
      }

      unsigned int versionSUBR = 0;
      if ( wantedDir->first->doTheseRegionsForm( reg.id, wantedDir->second.begin(), wantedDir->second.end(), versionSUBR ) ) {
         if ( versionHIT < versionSUBR && versionSUPER < versionSUBR ) {
            for ( std::map< reg_t, unsigned int >::const_iterator it = wantedDir->second.begin(); it != wantedDir->second.end(); it++ ) {
               global_reg_t r( it->first, wantedDir->first );
               reg_t intersect = r.key->computeIntersect( reg.id, r.id );
               if ( it->first == intersect ) {
                  locations.push_back( std::make_pair( it->first, it->first ) );
               }
            }
            NewNewDirectoryEntryData *firstEntry = ( NewNewDirectoryEntryData * ) wantedDir->first->getRegionData( reg.id );
            if ( firstEntry == NULL ) {
               firstEntry = NEW NewNewDirectoryEntryData(  );
               firstEntry->addAccess( 0, 1 );
               wantedDir->first->setRegionData( reg.id, firstEntry );
            }
            resultSUBR = true;
            version = versionSUBR;
         }
      }
      if ( !resultSUBR && ( resultSUPER || resultHIT ) ) {
         if ( versionHIT >= versionSUPER ) {
            version = versionHIT;
            locations.push_back( std::make_pair( reg.id, reg.id ) );
         } else {
            version = versionSUPER;
            locations.push_back( std::make_pair( reg.id, superPart ) );
            NewNewDirectoryEntryData *firstEntry = ( NewNewDirectoryEntryData * ) wantedDir->first->getRegionData( reg.id );
            NewNewDirectoryEntryData *secondEntry = ( NewNewDirectoryEntryData * ) wantedDir->first->getRegionData( superPart );
            if ( firstEntry == NULL ) {
               firstEntry = NEW NewNewDirectoryEntryData( *secondEntry );
               wantedDir->first->setRegionData( reg.id, firstEntry );
            } else {
               if (secondEntry == NULL) std::cerr << "LOLWTF!"<< std::endl;
               *firstEntry = *secondEntry;
            }
         }
      }
   }
   return (resultSUBR || resultSUPER || resultHIT) ;
}

void MemController::preInit( ) {
   unsigned int index;
   for ( index = 0; index < _wd.getNumCopies(); index += 1 ) {
      //std::cerr << "WD "<< _wd.getId() << " Depth: "<< _wd.getDepth() <<" Creating copy "<< index << std::endl;
      //std::cerr << _wd.getCopies()[ index ];
      new ( &_memCacheCopies[ index ] ) MemCacheCopy( _wd, index );
      hasVersionInfoForRegion( _memCacheCopies[ index ]._reg  , _memCacheCopies[ index ]._version, _memCacheCopies[ index ]._locations );
   }
}


void MemController::copyDataIn( memory_space_id_t destination ) {
   _memorySpaceId = destination;
   if ( _memorySpaceId == 0 /* HOST_MEMSPACE_ID */) {
      _inOps = NEW HostAddressSpaceInOps();
   } else {
      _inOps = NEW SeparateAddressSpaceInOps( sys.getSeparateMemory( _memorySpaceId ) );
   }
   
 std::cerr << "### copyDataIn wd " << _wd.getId() << std::endl; 
   for ( unsigned int index = 0; index < _wd.getNumCopies(); index++ ) {
      _memCacheCopies[ index ].getVersionInfo();
      std::cerr << "## "; _memCacheCopies[ index ]._reg.key->printRegion( _memCacheCopies[ index ]._reg.id ); std::cerr << std::endl;
   }
   
   for ( unsigned int index = 0; index < _wd.getNumCopies(); index++ ) {
      _memCacheCopies[ index ].generateInOps2( *_inOps, _wd.getCopies()[index].isInput(), _wd.getCopies()[index].isOutput(), _wd );
   }

   _inOps->issue( _wd );
 std::cerr << "### copyDataIn wd " << _wd.getId() << " done" << std::endl;
}

void MemController::copyDataOut( ) {
   if ( _memorySpaceId == 0 /* HOST_MEMSPACE_ID */) {
   } else {
      _outOps = NEW SeparateAddressSpaceOutOps( sys.getSeparateMemory( _memorySpaceId ) );

      for ( unsigned int index = 0; index < _wd.getNumCopies(); index++ ) {
         _memCacheCopies[ index ].generateOutOps( *_outOps, _wd.getCopies()[index].isInput(), _wd.getCopies()[index].isOutput() );
      }

      _outOps->issue( _wd, _memCacheCopies );
   }
}

uint64_t MemController::getAddress( unsigned int index ) const {
   uint64_t addr = 0;
   //std::cerr << " _getAddress, reg: " << index << " key: " << (void *)_memCacheCopies[ index ]._reg.key << " id: " << _memCacheCopies[ index ]._reg.id << std::endl;
   if ( _memorySpaceId == 0 ) {
      addr = ((uint64_t) _wd.getCopies()[ index ].getBaseAddress()) + _wd.getCopies()[ index ].getOffset() ;
   } else {
      addr = sys.getSeparateMemory( _memorySpaceId ).getDeviceAddress( _memCacheCopies[ index ]._reg, (uint64_t) _wd.getCopies()[ index ].getBaseAddress() );
   }
   return addr;
}

void MemController::getInfoFromPredecessor( MemController const &predecessorController ) {
   //std::cerr << _wd.getId() <<" checking predecessor info from " << predecessorController._wd.getId() << std::endl;
   _provideLock.acquire();
   for( unsigned int index = 0; index < predecessorController._wd.getNumCopies(); index += 1) {
      std::map< reg_t, unsigned int > &regs = _providedRegions[ predecessorController._memCacheCopies[ index ]._reg.key ];
      regs[ predecessorController._memCacheCopies[ index ]._reg.id ] = ( ( predecessorController._wd.getCopies()[index].isOutput() ) ? predecessorController._memCacheCopies[ index ]._version + 1 : predecessorController._memCacheCopies[ index ]._version );
      //std::cerr << "provided data for copy " << index << " reg ("<<predecessorController._cacheCopies[ index ]._reg.key<<"," << predecessorController._cacheCopies[ index ]._reg.id << ") with version " << ( ( predecessorController._cacheCopies[index].getCopyData().isOutput() ) ? predecessorController._cacheCopies[ index ].getNewVersion() + 1 : predecessorController._cacheCopies[ index ].getNewVersion() ) << " isOut "<< predecessorController._cacheCopies[index].getCopyData().isOutput()<< " isIn "<< predecessorController._cacheCopies[index].getCopyData().isInput() << std::endl;
   }
   _provideLock.release();
}
 
bool MemController::isDataReady() {
   return _inOps->isDataReady();
}

}
