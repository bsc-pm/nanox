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
MemController::MemController( WD &wd ) : _initialized( false ), _preinitialized(false), _inputDataReady(false),
      _outputDataReady(false), _memoryAllocated( false ),
      _mainWd( false ), _wd( wd ), _pe( NULL ), _provideLock(), _providedRegions(), _affinityScore( 0 ),
      _maxAffinityScore( 0 ), _ownedRegions(), _parentRegions() {
   if ( _wd.getNumCopies() > 0 ) {
      _memCacheCopies = NEW MemCacheCopy[ wd.getNumCopies() ];
   }
}

#if 0
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
         //double check the directory because a there may be WDs that have not been detected as predecessors
         NewNewDirectoryEntryData *entry = ( NewNewDirectoryEntryData * ) wantedDir->first->getRegionData( wantedReg->first );
         if ( entry->getVersion() > versionHIT ) {
            versionHIT = entry->getVersion();
         }
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
#endif

bool MemController::ownsRegion( global_reg_t const &reg ) {
   bool i_has_it = _ownedRegions.hasObjectOfRegion( reg );
   bool parent_has_it  = _parentRegions.hasObjectOfRegion( reg );
   //std::cerr << " wd: " << _wd.getId() << " i has it? " << (i_has_it ? "yes" : "no") << " " << &_ownedRegions << ", parent has it? " << (parent_has_it ? "yes" : "no") << " " << &_parentRegions << std::endl;
   return i_has_it || parent_has_it;
}

void MemController::preInit( ) {
   unsigned int index;
   if ( _preinitialized ) return;
   if ( _VERBOSE_CACHE ) { 
      *(myThread->_file) << " (preinit)INITIALIZING MEMCONTROLLER for WD " << _wd.getId() << " " << (_wd.getDescription()!=NULL ? _wd.getDescription() : "n/a")  << " NUM COPIES " << _wd.getNumCopies() << std::endl;
   }

   //std::ostream &o = (*myThread->_file);
   //o << "### preInit wd " << _wd.getId() << std::endl;
   for ( index = 0; index < _wd.getNumCopies(); index += 1 ) {
      //std::cerr << "WD "<< _wd.getId() << " Depth: "<< _wd.getDepth() <<" Creating copy "<< index << std::endl;
      //std::cerr << _wd.getCopies()[ index ];
      //
      //
      //

   uint64_t host_copy_addr = 0;
   if ( _wd.getParent() != NULL /* && !_wd.getParent()->_mcontrol._mainWd */ ) {
      for ( unsigned int parent_idx = 0; parent_idx < _wd.getParent()->getNumCopies(); parent_idx += 1 ) {
         if ( _wd.getParent()->_mcontrol.getAddress( parent_idx ) == (uint64_t) _wd.getCopies()[ index ].getBaseAddress() ) {
            host_copy_addr = (uint64_t) _wd.getParent()->getCopies()[ parent_idx ].getHostBaseAddress();
            //std::cerr << "TADAAAA this comes from a father's copy "<< std::hex << host_copy_addr << std::endl;
            _wd.getCopies()[ index ].setHostBaseAddress( host_copy_addr );
         }
      }
   }
      new ( &_memCacheCopies[ index ] ) MemCacheCopy( _wd, index );

      // o << "## " << (_wd.getCopies()[index].isInput() ? "in" : "") << (_wd.getCopies()[index].isOutput() ? "out" : "") << " " <<  _wd.getCopies()[index] << std::endl; 

      unsigned int predecessorsVersion;
      if ( _providedRegions.hasVersionInfoForRegion( _memCacheCopies[ index ]._reg, predecessorsVersion, _memCacheCopies[ index ]._locations ) ) {
         _memCacheCopies[ index ].setVersion( predecessorsVersion );
      }
      if ( _memCacheCopies[ index ].getVersion() != 0 ) {
         if ( _VERBOSE_CACHE ) { *(myThread->_file) << "WD " << _wd.getId() << " copy "<< index <<" got location info from predecessor "<<  _memCacheCopies[ index ]._reg.id << " got version " << _memCacheCopies[ index ].getVersion()<< " "; }
         _memCacheCopies[ index ]._locationDataReady = true;
      } else {
         if ( _VERBOSE_CACHE ) { *(myThread->_file) << "WD " << _wd.getId() << " copy "<< index <<" got requesting location info to global directory for region "<<  _memCacheCopies[ index ]._reg.id << " "; }
         _memCacheCopies[ index ].getVersionInfo();
      }
      if ( _VERBOSE_CACHE ) { 
         for ( NewLocationInfoList::const_iterator it = _memCacheCopies[ index ]._locations.begin(); it != _memCacheCopies[ index ]._locations.end(); it++ ) {
               NewNewDirectoryEntryData *rsentry = ( NewNewDirectoryEntryData * ) _memCacheCopies[ index ]._reg.key->getRegionData( it->first );
               NewNewDirectoryEntryData *dsentry = ( NewNewDirectoryEntryData * ) _memCacheCopies[ index ]._reg.key->getRegionData( it->second );
            *(myThread->_file) << "<" << it->first << ": [" << *rsentry << "] ," << it->second << " : [" << *dsentry << "] > ";
         }
         *(myThread->_file) << std::endl;
      }

      if ( _wd.getParent() != NULL && _wd.getParent()->_mcontrol.ownsRegion( _memCacheCopies[ index ]._reg ) ) {
         /* do nothing, maybe here we can add a correctness check,
          * to ensure that the region is a subset of the Parent regions
          */
         //std::cerr << "I am " << _wd.getId() << " parent: " <<  _wd.getParent()->getId() << " NOT ADDING THIS OBJECT "; _memCacheCopies[ index ]._reg.key->printRegion(std::cerr, 1); std::cerr << " adding it to " << &_parentRegions << std::endl;
         _parentRegions.addRegion( _memCacheCopies[ index ]._reg, _memCacheCopies[ index ].getVersion() );
      } else { /* this should be for private data */
         if ( _wd.getParent() != NULL ) {
         //std::cerr << "I am " << _wd.getId() << " parent: " << _wd.getParent()->getId() << " ++++++ ADDING THIS OBJECT "; _memCacheCopies[ index ]._reg.key->printRegion(std::cerr, 1); std::cerr << std::endl;
            _wd.getParent()->_mcontrol._ownedRegions.addRegion( _memCacheCopies[ index ]._reg, _memCacheCopies[ index ].getVersion() );
         }
      }
   }



   //  if ( _VERBOSE_CACHE ) {
   //     //std::ostream &o = (*myThread->_file);
   //     o << "### preInit wd " << _wd.getId() << std::endl;
   //     for ( index = 0; index < _wd.getNumCopies(); index += 1 ) {
   //        o << "## " << (_wd.getCopies()[index].isInput() ? "in" : "") << (_wd.getCopies()[index].isOutput() ? "out" : "") << " "; _memCacheCopies[ index ]._reg.key->printRegion( o, _memCacheCopies[ index ]._reg.id ) ;
   //        o << std::endl; 
   //     }
   //  }




   if ( _VERBOSE_CACHE ) { 
      *(myThread->_file) << " (preinit)END OF INITIALIZING MEMCONTROLLER for WD " << _wd.getId() << " " << (_wd.getDescription()!=NULL ? _wd.getDescription() : "n/a")  << " NUM COPIES " << _wd.getNumCopies() << " &_preinitialized= "<< &_preinitialized<< std::endl;
   }
   _preinitialized = true;
}

void MemController::initialize( ProcessingElement &pe ) {
   if ( !_initialized ) {
      _pe = &pe;
      //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_CC_CDIN); );

      if ( _pe->getMemorySpaceId() == 0 /* HOST_MEMSPACE_ID */) {
         _inOps = NEW HostAddressSpaceInOps( _pe, true );
      } else {
         _inOps = NEW SeparateAddressSpaceInOps( _pe, true, sys.getSeparateMemory( _pe->getMemorySpaceId() ) );
      }
      _initialized = true;
   } else {
      ensure(_pe == &pe, " MemController, called initialize twice with different PE!");
   }
}

bool MemController::allocateTaskMemory() {
   bool result = true;
   if ( _pe->getMemorySpaceId() != 0 ) {
      result = sys.getSeparateMemory( _pe->getMemorySpaceId() ).prepareRegions( _memCacheCopies, _wd.getNumCopies(), _wd );
   }
   if ( result ) {
      //*(myThread->_file) << "++++ Succeeded allocation for wd " << _wd.getId() << std::endl;
      for ( unsigned int idx = 0; idx < _wd.getNumCopies(); idx += 1 ) {
         if ( _memCacheCopies[idx]._reg.key->getKeepAtOrigin() ) {
            //std::cerr << "WD " << _wd.getId() << " rooting to memory space " << _pe->getMemorySpaceId() << std::endl;
            _memCacheCopies[idx]._reg.setOwnedMemory( _pe->getMemorySpaceId() );
         }
      }
   }
   _memoryAllocated = result;
   return result;
}

void MemController::copyDataIn() {
   ensure( _preinitialized == true, "MemController not initialized!");
   ensure( _initialized == true, "MemController not initialized!");
  
   if ( _VERBOSE_CACHE || sys.getVerboseCopies() ) {
      //if ( sys.getNetwork()->getNodeNum() == 0 ) {
         std::ostream &o = (*myThread->_file);
         o << "### copyDataIn wd " << std::dec << _wd.getId() << " (" << (_wd.getDescription()!=NULL?_wd.getDescription():"[no desc]")<< ") running on " << std::dec << _pe->getMemorySpaceId() << " ops: "<< (void *) _inOps << std::endl;
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
   if ( _VERBOSE_CACHE || sys.getVerboseCopies() ) {
      if ( sys.getNetwork()->getNodeNum() == 0 ) {
         (*myThread->_file) << "### copyDataIn wd " << std::dec << _wd.getId() << " done" << std::endl;
      }
   }
   //NANOS_INSTRUMENT( inst2.close(); );
}

void MemController::copyDataOut( MemControllerPolicy policy ) {
   ensure( _preinitialized == true, "MemController not initialized!");
   ensure( _initialized == true, "MemController not initialized!");

   //for ( unsigned int index = 0; index < _wd.getNumCopies(); index++ ) {
   //   if ( _wd.getCopies()[index].isInput() && _wd.getCopies()[index].isOutput() ) {
   //      _memCacheCopies[ index ]._reg.setLocationAndVersion( _pe->getMemorySpaceId(), _memCacheCopies[ index ].getVersion() + 1 );
   //   }
   //}
   if ( _VERBOSE_CACHE || sys.getVerboseCopies() ) { *(myThread->_file) << "### copyDataOut wd " << std::dec << _wd.getId() << " metadata set, not released yet" << std::endl; }


   for ( unsigned int index = 0; index < _wd.getNumCopies(); index++ ) {
      if ( _wd.getCopies()[index].isOutput() ) {
         if ( _wd.getParent() != NULL && _wd.getParent()->_mcontrol.ownsRegion( _memCacheCopies[index]._reg ) ) {
            WD &parent = *(_wd.getParent());
            for ( unsigned int parent_idx = 0; parent_idx < parent.getNumCopies(); parent_idx += 1) {
               if ( parent._mcontrol._memCacheCopies[parent_idx]._reg.contains( _memCacheCopies[ index ]._reg ) ) {
                  if ( parent._mcontrol._memCacheCopies[parent_idx].getChildrenProducedVersion() < _memCacheCopies[ index ].getChildrenProducedVersion() ) {
                     parent._mcontrol._memCacheCopies[parent_idx].setChildrenProducedVersion( _memCacheCopies[ index ].getChildrenProducedVersion() );
                  }
               }
            }
         }
      }
   }



   if ( _pe->getMemorySpaceId() == 0 /* HOST_MEMSPACE_ID */) {
      _outputDataReady = true;
   } else {
      _outOps = NEW SeparateAddressSpaceOutOps( _pe, false, true );

      for ( unsigned int index = 0; index < _wd.getNumCopies(); index++ ) {
         _memCacheCopies[ index ].generateOutOps( &sys.getSeparateMemory( _pe->getMemorySpaceId() ), *_outOps, _wd.getCopies()[index].isInput(), _wd.getCopies()[index].isOutput(), _wd, index );
      }

      //if( sys.getNetwork()->getNodeNum()== 0)std::cerr << "MemController::copyDataOut for wd " << _wd.getId() << std::endl;
      _outOps->issue( _wd );
   }
}

uint64_t MemController::getAddress( unsigned int index ) const {
   ensure( _preinitialized == true, "MemController not preinitialized!");
   ensure( _initialized == true, "MemController not initialized!");
   uint64_t addr = 0;
   //std::cerr << " _getAddress, reg: " << index << " key: " << (void *)_memCacheCopies[ index ]._reg.key << " id: " << _memCacheCopies[ index ]._reg.id << std::endl;
   if ( _pe->getMemorySpaceId() == 0 ) {
      addr = ((uint64_t) _wd.getCopies()[ index ].getBaseAddress());
   } else {
      addr = sys.getSeparateMemory( _pe->getMemorySpaceId() ).getDeviceAddress( _memCacheCopies[ index ]._reg, (uint64_t) _wd.getCopies()[ index ].getBaseAddress(), _memCacheCopies[ index ]._chunk );
      //std::cerr << "Hola: HostBaseAddr: " << (void*) _wd.getCopies()[ index ].getHostBaseAddress() << " BaseAddr: " << (void*)_wd.getCopies()[ index ].getBaseAddress() << std::endl;
      //if ( _wd.getCopies()[ index ].isRemoteHost() || _wd.getCopies()[ index ].getHostBaseAddress() == 0 ) {
      //   std::cerr << "Hola" << std::endl;
      //   addr = sys.getSeparateMemory( _pe->getMemorySpaceId() ).getDeviceAddress( _memCacheCopies[ index ]._reg, (uint64_t) _wd.getCopies()[ index ].getBaseAddress(), _memCacheCopies[ index ]._chunk );
      //} else {
      //   std::cerr << "Hola1" << std::endl;
      //   addr = sys.getSeparateMemory( _pe->getMemorySpaceId() ).getDeviceAddress( _memCacheCopies[ index ]._reg, (uint64_t) _wd.getCopies()[ index ].getHostBaseAddress(), _memCacheCopies[ index ]._chunk );
      //}
   }
   //std::cerr << "MemController::getAddress " << (_wd.getDescription()!=NULL?_wd.getDescription():"[no desc]") <<" index: " << index << " addr: " << (void *)addr << std::endl;
   return addr;
}

void MemController::getInfoFromPredecessor( MemController const &predecessorController ) {
   for( unsigned int index = 0; index < predecessorController._wd.getNumCopies(); index += 1) {
      unsigned int version = predecessorController._memCacheCopies[ index ].getChildrenProducedVersion(); 
      unsigned int predecessorProducedVersion = predecessorController._memCacheCopies[ index ].getVersion() + (predecessorController._wd.getCopies()[ index ].isOutput() ? 1 : 0);
      if ( predecessorProducedVersion == version ) {
         // if the predecessor's children produced new data, then the father can not
         // guarantee that the version is correct (the children may have produced a subchunk
         // of the region). The version is not added here and then the global directory is checked.
         //(*myThread->_file) << "getInfoFromPredecessor[ " << _wd.getId() << " : "<< _wd.getDescription()<< " key: " << (void*)predecessorController._memCacheCopies[ index ]._reg.key << " ] adding version " << version << " from wd " << predecessorController._wd.getId() << " : " << predecessorController._wd.getDescription() << " : " << index << " copy version " << predecessorController._memCacheCopies[ index ].getVersion() << std::endl;
         _providedRegions.addRegion( predecessorController._memCacheCopies[ index ]._reg, version );
      }
   }
#if 0
   _provideLock.acquire();
   for( unsigned int index = 0; index < predecessorController._wd.getNumCopies(); index += 1) {
      std::map< reg_t, unsigned int > &regs = _providedRegions[ predecessorController._memCacheCopies[ index ]._reg.key ];
      std::map< reg_t, unsigned int >::iterator elem = regs.find( predecessorController._memCacheCopies[ index ]._reg.id );

      unsigned int version = predecessorController._memCacheCopies[ index ].getVersion() + ( predecessorController._wd.getCopies()[index].isOutput() ? 1 : 0 );
      //(*myThread->_file) << "getInfoFromPredecessor[ " << _wd.getId() << " : "<< _wd.getDescription()<< " key: " << (void*)predecessorController._memCacheCopies[ index ]._reg.key << " ] adding version " << version << " from wd " << predecessorController._wd.getId() << " : " << predecessorController._wd.getDescription() << " : " << index << std::endl;
      if ( elem != regs.end() ) {
         if ( elem->second < version ) {
            regs[ elem->first ] = version;
         }
      } else {
         regs[ predecessorController._memCacheCopies[ index ]._reg.id ] = version;
      }
      //std::cerr << "from wd " << predecessorController._wd.getId() << " to wd " << _wd.getId()  << " provided data for copy " << index << " reg ("<<predecessorController._memCacheCopies[ index ]._reg.key<<"," << predecessorController._memCacheCopies[ index ]._reg.id << ") with version " << ( ( predecessorController._wd.getCopies()[index].isOutput() ) ? predecessorController._memCacheCopies[ index ].getVersion() + 1 : predecessorController._memCacheCopies[ index ].getVersion() ) << " isOut "<< predecessorController._wd.getCopies()[index].isOutput()<< " isIn "<< predecessorController._wd.getCopies()[index].isInput() << std::endl;
   }
   _provideLock.release();
#endif
}
 
bool MemController::isDataReady( WD const &wd ) {
   ensure( _preinitialized == true, "MemController not initialized!");
   if ( _initialized ) {
      if ( !_inputDataReady ) {
         _inputDataReady = _inOps->isDataReady( wd );
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
            if ( _VERBOSE_CACHE ) { *(myThread->_file) << "Output data is ready for wd " << _wd.getId() << " obj " << (void *)_outOps << std::endl; }

            sys.getSeparateMemory( _pe->getMemorySpaceId() ).releaseRegions( _memCacheCopies, _wd.getNumCopies(), _wd ) ;
            //for ( unsigned int index = 0; index < _wd.getNumCopies(); index++ ) {
            //   sys.getSeparateMemory( _pe->getMemorySpaceId() ).releaseRegions( _memCacheCopies, _wd.getNumCopies(), _wd ) ;
            //}
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
   memory_space_id_t refLoc = (memory_space_id_t) -1;
   for ( unsigned int index = 0; index < _wd.getNumCopies(); index++ ) {
      memory_space_id_t thisLoc;
      if ( _memCacheCopies[ index ].isRooted( thisLoc ) ) {
         thisLoc = thisLoc == 0 ? 0 : ( sys.getSeparateMemory( thisLoc ).getNodeNumber() != 0 ? thisLoc : 0 );
         if ( refLoc == (memory_space_id_t) -1 ) {
            refLoc = thisLoc;
            result = true;
         } else {
            result = (refLoc == thisLoc);
         }
      }
   }
   if ( result ) loc = refLoc;
   return result;
}

void MemController::setMainWD() {
   _mainWd = true;
}

void MemController::synchronize() {
   sys.getHostMemory().synchronize( _wd );
}

bool MemController::isMemoryAllocated() const {
   return _memoryAllocated;
}

void MemController::setCacheMetaData() {
   for ( unsigned int index = 0; index < _wd.getNumCopies(); index++ ) {
      if ( _wd.getCopies()[index].isOutput() ) {
         unsigned int newVersion = _memCacheCopies[ index ].getVersion() + 1;
         _memCacheCopies[ index ]._reg.setLocationAndVersion( _pe, _pe->getMemorySpaceId(), newVersion ); //update directory
         _memCacheCopies[ index ].setChildrenProducedVersion( newVersion );
         if ( _pe->getMemorySpaceId() != 0 /* HOST_MEMSPACE_ID */) {
            sys.getSeparateMemory( _pe->getMemorySpaceId() ).setRegionVersion( _memCacheCopies[ index ]._reg, _memCacheCopies[ index ].getVersion() + 1, _wd, index );
         }
      } else if ( _wd.getCopies()[index].isInput() ) {
         _memCacheCopies[ index ].setChildrenProducedVersion( _memCacheCopies[ index ].getVersion() );
      }
   }
}

bool MemController::hasObjectOfRegion( global_reg_t const &reg ) {
   return _ownedRegions.hasObjectOfRegion( reg );
}

}
