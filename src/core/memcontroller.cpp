#include "memcontroller_decl.hpp"
#include "workdescriptor.hpp"
#include "regiondict.hpp"
#include "newregiondirectory.hpp"

#if VERBOSE_CACHE
 #define _VERBOSE_CACHE 1
#else
 #define _VERBOSE_CACHE 0
#endif

namespace nanos {
MemController::MemController( WD const &wd ) : _initialized( false ), _wd( wd ), _memorySpaceId( 0 ), _provideLock(), _providedRegions(), _affinityScore( 0 ), _maxAffinityScore( 0 )  {
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
            NewNewDirectoryEntryData *dirEntry = ( NewNewDirectoryEntryData * ) wantedDir->first->getRegionData( reg.id );
            if ( dirEntry != NULL ) { /* if entry is null, do check directory, because we need to insert the region info in the intersect maps */
               for ( std::map< reg_t, unsigned int >::const_iterator it = wantedDir->second.begin(); it != wantedDir->second.end(); it++ ) {
                  global_reg_t r( it->first, wantedDir->first );
                  reg_t intersect = r.key->computeIntersect( reg.id, r.id );
                  if ( it->first == intersect ) {
                     locations.push_back( std::make_pair( it->first, it->first ) );
                  }
               }
               version = versionSUBR;
            } else {
               sys.getHostMemory().getVersionInfo( reg, version, locations );
            }
            resultSUBR = true;
            //std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! VERSION INFO !!! CHUNKS FORM THIS REG!!! and version computed is " << version << std::endl;
         }
      }
      if ( !resultSUBR && ( resultSUPER || resultHIT ) ) {
         if ( versionHIT >= versionSUPER ) {
            version = versionHIT;
            //std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! VERSION INFO !!! CHUNKS HIT!!! and version computed is " << version << std::endl;
            locations.push_back( std::make_pair( reg.id, reg.id ) );
         } else {
            version = versionSUPER;
            //std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! VERSION INFO !!! CHUNKS COMES FROM A BIGGER!!! and version computed is " << version << std::endl;
            NewNewDirectoryEntryData *firstEntry = ( NewNewDirectoryEntryData * ) wantedDir->first->getRegionData( reg.id );
            if ( firstEntry != NULL ) {
               locations.push_back( std::make_pair( reg.id, superPart ) );
               NewNewDirectoryEntryData *secondEntry = ( NewNewDirectoryEntryData * ) wantedDir->first->getRegionData( superPart );
               if (secondEntry == NULL) std::cerr << "LOLWTF!"<< std::endl;
               *firstEntry = *secondEntry;
            } else {
               sys.getHostMemory().getVersionInfo( reg, version, locations );
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
      if ( _memCacheCopies[ index ]._version != 0 ) {
         if ( _VERBOSE_CACHE ) { std::cerr << "WD " << _wd.getId() << " copy "<< index <<" got location info from predecessor "<<  _memCacheCopies[ index ]._reg.id << " "; }
         _memCacheCopies[ index ]._locationDataReady = true;
      } else {
         if ( _VERBOSE_CACHE ) { std::cerr << "WD " << _wd.getId() << " copy "<< index <<" got requesting location info to global directory for region "<<  _memCacheCopies[ index ]._reg.id << " "; }
         _memCacheCopies[ index ].getVersionInfo();
      }
      if ( _VERBOSE_CACHE ) { 
         for ( NewLocationInfoList::const_iterator it = _memCacheCopies[ index ]._locations.begin(); it != _memCacheCopies[ index ]._locations.end(); it++ ) {
               NewNewDirectoryEntryData *rsentry = ( NewNewDirectoryEntryData * ) _memCacheCopies[ index ]._reg.key->getRegionData( it->first );
               NewNewDirectoryEntryData *dsentry = ( NewNewDirectoryEntryData * ) _memCacheCopies[ index ]._reg.key->getRegionData( it->second );
            std::cerr << "<" << it->first << ": [" << *rsentry << "] ," << it->second << " : [" << *dsentry << "] > ";
         }
         std::cerr << std::endl;
      }
   }
}

void MemController::initialize( ProcessingElement &pe ) {
   if ( !_initialized ) {
      _memorySpaceId = pe.getMemorySpaceId();
      //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_CC_CDIN); );

      if ( _memorySpaceId == 0 /* HOST_MEMSPACE_ID */) {
         _inOps = NEW HostAddressSpaceInOps( true );
      } else {
         _inOps = NEW SeparateAddressSpaceInOps( true, sys.getSeparateMemory( _memorySpaceId ) );
      }
      _initialized = true;
   }
}

bool MemController::allocateInputMemory() {
   ensure( _inOps != NULL, "NULL ops." );
   return _inOps->prepareRegions( _memCacheCopies, _wd.getNumCopies(), _wd );
}

void MemController::copyDataIn() {
   
   if ( _VERBOSE_CACHE ) {
      if ( sys.getNetwork()->getNodeNum() == 0 ) {
         std::cerr << "### copyDataIn wd " << _wd.getId() << " running on " << _memorySpaceId << " ops: "<< (void *) _inOps << std::endl;
         for ( unsigned int index = 0; index < _wd.getNumCopies(); index++ ) {
         NewNewDirectoryEntryData *d = NewNewRegionDirectory::getDirectoryEntry( *(_memCacheCopies[ index ]._reg.key), _memCacheCopies[ index ]._reg.id );
         std::cerr << "## "; _memCacheCopies[ index ]._reg.key->printRegion( _memCacheCopies[ index ]._reg.id ) ;
         if ( d ) std::cerr << " " << *d << std::endl; 
         else std::cerr << " dir entry n/a" << std::endl;
         }
      }
   }
   
   //if( sys.getNetwork()->getNodeNum()== 0)std::cerr << "MemController::copyDataIn for wd " << _wd.getId() << std::endl;
   for ( unsigned int index = 0; index < _wd.getNumCopies(); index++ ) {
      _memCacheCopies[ index ].generateInOps( *_inOps, _wd.getCopies()[index].isInput(), _wd.getCopies()[index].isOutput(), _wd );
   }

   //NANOS_INSTRUMENT( InstrumentState inst5(NANOS_CC_CDIN_DO_OP); );
   _inOps->issue( _wd );
   //NANOS_INSTRUMENT( inst5.close(); );
   if ( _VERBOSE_CACHE ) {
      if ( sys.getNetwork()->getNodeNum() == 0 ) {
         std::cerr << "### copyDataIn wd " << _wd.getId() << " done" << std::endl;
      }
   }
   //NANOS_INSTRUMENT( inst2.close(); );
}

void MemController::copyDataOut( ) {

   for ( unsigned int index = 0; index < _wd.getNumCopies(); index++ ) {
      if ( _wd.getCopies()[index].isOutput() ) {
         _memCacheCopies[ index ]._reg.setLocationAndVersion( _memorySpaceId, _memCacheCopies[ index ]._version + 1 );
      }
   }
   if ( _VERBOSE_CACHE ) { std::cerr << "### copyDataOut wd " << _wd.getId() << " metadata set, not released yet" << std::endl; }


   if ( _memorySpaceId == 0 /* HOST_MEMSPACE_ID */) {
   } else {
      _outOps = NEW SeparateAddressSpaceOutOps( false, false );

      for ( unsigned int index = 0; index < _wd.getNumCopies(); index++ ) {
         _memCacheCopies[ index ].generateOutOps( *_outOps, _wd.getCopies()[index].isInput(), _wd.getCopies()[index].isOutput() );
      }

      //if( sys.getNetwork()->getNodeNum()== 0)std::cerr << "MemController::copyDataOut for wd " << _wd.getId() << std::endl;
      _outOps->issue( _wd );

      for ( unsigned int index = 0; index < _wd.getNumCopies(); index++ ) {
         sys.getSeparateMemory( _memorySpaceId ).releaseRegion( _memCacheCopies[ index ]._reg, _wd ) ;
      }
   }
}

uint64_t MemController::getAddress( unsigned int index ) const {
   uint64_t addr = 0;
   //std::cerr << " _getAddress, reg: " << index << " key: " << (void *)_memCacheCopies[ index ]._reg.key << " id: " << _memCacheCopies[ index ]._reg.id << std::endl;
   if ( _memorySpaceId == 0 ) {
      addr = ((uint64_t) _wd.getCopies()[ index ].getBaseAddress());
   } else {
      addr = sys.getSeparateMemory( _memorySpaceId ).getDeviceAddress( _memCacheCopies[ index ]._reg, (uint64_t) _wd.getCopies()[ index ].getBaseAddress(), _memCacheCopies[ index ]._chunk );
   }
   return addr;
}

void MemController::getInfoFromPredecessor( MemController const &predecessorController ) {
   _provideLock.acquire();
   for( unsigned int index = 0; index < predecessorController._wd.getNumCopies(); index += 1) {
      std::map< reg_t, unsigned int > &regs = _providedRegions[ predecessorController._memCacheCopies[ index ]._reg.key ];
      regs[ predecessorController._memCacheCopies[ index ]._reg.id ] = ( ( predecessorController._wd.getCopies()[index].isOutput() ) ? predecessorController._memCacheCopies[ index ]._version + 1 : predecessorController._memCacheCopies[ index ]._version );
      //std::cerr << "provided data for copy " << index << " reg ("<<predecessorController._cacheCopies[ index ]._reg.key<<"," << predecessorController._cacheCopies[ index ]._reg.id << ") with version " << ( ( predecessorController._cacheCopies[index].getCopyData().isOutput() ) ? predecessorController._cacheCopies[ index ].getNewVersion() + 1 : predecessorController._cacheCopies[ index ].getNewVersion() ) << " isOut "<< predecessorController._cacheCopies[index].getCopyData().isOutput()<< " isIn "<< predecessorController._cacheCopies[index].getCopyData().isInput() << std::endl;
   }
   _provideLock.release();
}
 
bool MemController::isDataReady( WD const &wd ) {
   bool result = _inOps->isDataReady( wd );
   if ( result ) {
      if ( _VERBOSE_CACHE ) { std::cerr << "Data is ready for wd " << _wd.getId() << " obj " << (void *)_inOps << std::endl; }
      _inOps->releaseLockedSourceChunks();
   }
   return result;
}

bool MemController::canAllocateMemory( memory_space_id_t memId, bool considerInvalidations ) const {
   if ( memId > 0 ) {
      return sys.getSeparateMemory( memId ).canAllocateMemory( _memCacheCopies, _wd.getNumCopies(), considerInvalidations );
   } else {
      return true;
   }
}


void MemController::setAffinityScore( std::size_t score ) {
   _affinityScore = score;
}

std::size_t MemController::getAffinityScore() const {
   return _affinityScore;
}

void MemController::setMaxAffinityScore( std::size_t score ) {
   _maxAffinityScore = score;
}

std::size_t MemController::getMaxAffinityScore() const {
   return _maxAffinityScore;
}

std::size_t MemController::getAmountOfTransferredData() const {
   return ( _inOps != NULL ) ? _inOps->getAmountOfTransferredData() : 0 ;
}

std::size_t MemController::getTotalAmountOfData() const {
   std::size_t total = 0;
   for ( unsigned int index = 0; index < _wd.getNumCopies(); index++ ) {
      total += _memCacheCopies[ index ]._reg.getDataSize();
   }
   return total;
}

}
