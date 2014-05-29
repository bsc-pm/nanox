#include "memcontroller_decl.hpp"
#include "workdescriptor.hpp"
#include "regiondict.hpp"
#include "newregiondirectory.hpp"

#if VERBOSE_CACHE
 #define _VERBOSE_CACHE 1
#else
 #define _VERBOSE_CACHE 0
 //#define _VERBOSE_CACHE ( sys.getNetwork()->getNodeNum() == 0 )
#endif

namespace nanos {
MemController::MemController( WD const &wd ) : _initialized( false ), _preinitialized(false), _inputDataReady(false),
      _outputDataReady(false), _memoryAllocated( false ),
      _mainWd( false ), _wd( wd ), _memorySpaceId( 0 ), _provideLock(), _providedRegions(), _affinityScore( 0 ),
      _maxAffinityScore( 0 )  {
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
      //if ( resultHIT ) {
      //   std::cerr << " HIT got version " << versionHIT << " for region " << reg.id << std::endl;
      //}

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
   if ( _preinitialized ) return;
   if ( _VERBOSE_CACHE ) { 
      *(myThread->_file) << " (preinit)INITIALIZING MEMCONTROLLER for WD " << _wd.getId() << " " << (_wd.getDescription()!=NULL ? _wd.getDescription() : "n/a")  << " NUM COPIES " << _wd.getNumCopies() << std::endl;
   }

   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t debug_key = ID->getEventKey("debug"); )

   for ( index = 0; index < _wd.getNumCopies(); index += 1 ) {
      //std::cerr << "WD "<< _wd.getId() << " Depth: "<< _wd.getDepth() <<" Creating copy "<< index << std::endl;
      //std::cerr << _wd.getCopies()[ index ];
   NANOS_INSTRUMENT( nanos_event_value_t debug_val_first = 0; )
   NANOS_INSTRUMENT( nanos_event_value_t debug_val_last = (_wd.getId() << 8) | ( (index+1) << 4 ) | 1; )
   NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents( 1, &debug_key, &debug_val_first ); )
      new ( &_memCacheCopies[ index ] ) MemCacheCopy( _wd, index );
      unsigned int predecessorsVersion;
      if ( hasVersionInfoForRegion( _memCacheCopies[ index ]._reg, predecessorsVersion, _memCacheCopies[ index ]._locations ) )
         _memCacheCopies[ index ].setVersion( predecessorsVersion );
      if ( _memCacheCopies[ index ].getVersion() != 0 ) {
         if ( _VERBOSE_CACHE ) { *(myThread->_file) << "WD " << _wd.getId() << " copy "<< index <<" got location info from predecessor "<<  _memCacheCopies[ index ]._reg.id << " "; }
         _memCacheCopies[ index ]._locationDataReady = true;
      } else {
         NANOS_INSTRUMENT( debug_val_last = (_wd.getId() << 8) | ( (index+1) << 4 ) | 2; )
         if ( _VERBOSE_CACHE ) { *(myThread->_file) << "WD " << _wd.getId() << " copy "<< index <<" got requesting location info to global directory for region "<<  _memCacheCopies[ index ]._reg.id << " "; }
         _memCacheCopies[ index ].getVersionInfo();
      }
   NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents( 1, &debug_key, &debug_val_last ); )
      if ( _VERBOSE_CACHE ) { 
         for ( NewLocationInfoList::const_iterator it = _memCacheCopies[ index ]._locations.begin(); it != _memCacheCopies[ index ]._locations.end(); it++ ) {
               NewNewDirectoryEntryData *rsentry = ( NewNewDirectoryEntryData * ) _memCacheCopies[ index ]._reg.key->getRegionData( it->first );
               NewNewDirectoryEntryData *dsentry = ( NewNewDirectoryEntryData * ) _memCacheCopies[ index ]._reg.key->getRegionData( it->second );
            *(myThread->_file) << "<" << it->first << ": [" << *rsentry << "] ," << it->second << " : [" << *dsentry << "] > ";
         }
         *(myThread->_file) << std::endl;
      }
   }


   if ( _VERBOSE_CACHE ) { 
      *(myThread->_file) << " (preinit)END OF INITIALIZING MEMCONTROLLER for WD " << _wd.getId() << " " << (_wd.getDescription()!=NULL ? _wd.getDescription() : "n/a")  << " NUM COPIES " << _wd.getNumCopies() << std::endl;
   }
   _preinitialized = true;
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
   bool result = _inOps->prepareRegions( _memCacheCopies, _wd.getNumCopies(), _wd );
   if ( result ) {
      for ( unsigned int idx = 0; idx < _wd.getNumCopies(); idx += 1 ) {
         if ( _memCacheCopies[idx]._reg.key->getKeepAtOrigin() ) {
            //std::cerr << "WD " << _wd.getId() << " rooting to memory space " << _memorySpaceId << std::endl;
            _memCacheCopies[idx]._reg.setRooted();
         }
      }
   }
   _memoryAllocated = result;
   return result;
}

void MemController::copyDataIn() {
   ensure( _preinitialized == true, "MemController not initialized!");
   ensure( _initialized == true, "MemController not initialized!");
  
   if ( _VERBOSE_CACHE ) {
      //if ( sys.getNetwork()->getNodeNum() == 0 ) {
         std::ostream &o = (*myThread->_file);
         o << "### copyDataIn wd " << _wd.getId() << " running on " << _memorySpaceId << " ops: "<< (void *) _inOps << std::endl;
         for ( unsigned int index = 0; index < _wd.getNumCopies(); index += 1 ) {
         NewNewDirectoryEntryData *d = NewNewRegionDirectory::getDirectoryEntry( *(_memCacheCopies[ index ]._reg.key), _memCacheCopies[ index ]._reg.id );
         o << "## " << (_wd.getCopies()[index].isInput() ? "in" : "") << (_wd.getCopies()[index].isOutput() ? "out" : "") << " "; _memCacheCopies[ index ]._reg.key->printRegion( o, _memCacheCopies[ index ]._reg.id ) ;
         if ( d ) o << " " << *d << std::endl; 
         else o << " dir entry n/a" << std::endl;
         _memCacheCopies[ index ].printLocations( o );
         }
      //}
   }
   
   //if( sys.getNetwork()->getNodeNum()== 0)std::cerr << "MemController::copyDataIn for wd " << _wd.getId() << std::endl;
   for ( unsigned int index = 0; index < _wd.getNumCopies(); index++ ) {
      _memCacheCopies[ index ].generateInOps( *_inOps, _wd.getCopies()[index].isInput(), _wd.getCopies()[index].isOutput(), _wd, index );
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

void MemController::copyDataOut( MemControllerPolicy policy ) {
   ensure( _preinitialized == true, "MemController not initialized!");
   ensure( _initialized == true, "MemController not initialized!");

   //for ( unsigned int index = 0; index < _wd.getNumCopies(); index++ ) {
   //   if ( _wd.getCopies()[index].isInput() && _wd.getCopies()[index].isOutput() ) {
   //      _memCacheCopies[ index ]._reg.setLocationAndVersion( _memorySpaceId, _memCacheCopies[ index ].getVersion() + 1 );
   //   }
   //}
   if ( _VERBOSE_CACHE ) { std::cerr << "### copyDataOut wd " << _wd.getId() << " metadata set, not released yet" << std::endl; }


   if ( _memorySpaceId == 0 /* HOST_MEMSPACE_ID */) {
      _outputDataReady = true;
   } else {
      _outOps = NEW SeparateAddressSpaceOutOps( false, false );

      for ( unsigned int index = 0; index < _wd.getNumCopies(); index++ ) {
         _memCacheCopies[ index ].generateOutOps( &sys.getSeparateMemory( _memorySpaceId ), *_outOps, _wd.getCopies()[index].isInput(), _wd.getCopies()[index].isOutput(), _wd, index );
      }

      //if( sys.getNetwork()->getNodeNum()== 0)std::cerr << "MemController::copyDataOut for wd " << _wd.getId() << std::endl;
      _outOps->issue( _wd );

   }
}

uint64_t MemController::getAddress( unsigned int index ) const {
   ensure( _preinitialized == true, "MemController not initialized!");
   ensure( _initialized == true, "MemController not initialized!");
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
      regs[ predecessorController._memCacheCopies[ index ]._reg.id ] = ( ( predecessorController._wd.getCopies()[index].isOutput() ) ? predecessorController._memCacheCopies[ index ].getVersion() + 1 : predecessorController._memCacheCopies[ index ].getVersion() );
      //std::cerr << "from wd " << predecessorController._wd.getId() << " to wd " << _wd.getId()  << " provided data for copy " << index << " reg ("<<predecessorController._memCacheCopies[ index ]._reg.key<<"," << predecessorController._memCacheCopies[ index ]._reg.id << ") with version " << ( ( predecessorController._wd.getCopies()[index].isOutput() ) ? predecessorController._memCacheCopies[ index ].getVersion() + 1 : predecessorController._memCacheCopies[ index ].getVersion() ) << " isOut "<< predecessorController._wd.getCopies()[index].isOutput()<< " isIn "<< predecessorController._wd.getCopies()[index].isInput() << std::endl;
   }
   _provideLock.release();
}
 
bool MemController::isDataReady( WD const &wd ) {
   ensure( _preinitialized == true, "MemController not initialized!");
   if ( _initialized ) {
      if ( !_inputDataReady ) {
         _inputDataReady = _inOps->isDataReady( wd );
         if ( _inputDataReady ) {
            if ( _VERBOSE_CACHE ) { std::cerr << "Data is ready for wd " << _wd.getId() << " obj " << (void *)_inOps << std::endl; }
            _inOps->releaseLockedSourceChunks();
         }
      }
      return _inputDataReady;
   } 
   return false;
}


bool MemController::isOutputDataReady( WD const &wd ) {
   ensure( _preinitialized == true, "MemController not initialized!");
   if ( _initialized ) {
      if ( !_outputDataReady ) {
         _outputDataReady = _outOps->isDataReady( wd );
         if ( _outputDataReady ) {
            if ( _VERBOSE_CACHE ) { std::cerr << "Output data is ready for wd " << _wd.getId() << " obj " << (void *)_outOps << std::endl; }

            for ( unsigned int index = 0; index < _wd.getNumCopies(); index++ ) {
               sys.getSeparateMemory( _memorySpaceId ).releaseRegion( _memCacheCopies[ index ]._reg, _wd, index, _memCacheCopies[ index ]._policy ) ;
            }
         }
      }
      return _outputDataReady;
   } 
   return false;
}

bool MemController::canAllocateMemory( memory_space_id_t memId, bool considerInvalidations ) const {
   if ( memId > 0 ) {
      return sys.getSeparateMemory( memId ).canAllocateMemory( _memCacheCopies, _wd.getNumCopies(), considerInvalidations, _wd );
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

bool MemController::isRooted( memory_space_id_t &loc ) const {
   bool result = false;
   unsigned int count = 0;
   for ( unsigned int index = 0; index < _wd.getNumCopies(); index++ ) {
      //std::cout << "Copy " << index << " addr " << (void *) _wd.getCopies()[index].getBaseAddress() << std::endl;
      if ( _memCacheCopies[ index ].isRooted( loc ) ) {
         count += 1;
         result = true; 
      }
      //std::cout << "Copy " << index << " addr " << (void *) _wd.getCopies()[index].getBaseAddress() << " count " << count << std::endl;
   }
   ensure(count <= 1, "Invalid count of rooted copies! (> 1).");
   return result;
}

void MemController::setMainWD() {
   _mainWd = true;
}

void MemController::synchronize() {
   if ( _mainWd ) {
      sys.getHostMemory().synchronize( _wd );
   } else {
 //     std::cerr << "Synchronize on a non main WD!, it will be considered as 'noflush'." << std::endl;
   }
}

bool MemController::isMemoryAllocated() const {
   return _memoryAllocated;
}

}
