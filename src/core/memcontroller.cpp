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

#include "memcontroller_decl.hpp"
#include "workdescriptor.hpp"
#include "regiondict.hpp"
#include "regiondirectory.hpp"
#include "memcachecopy.hpp"
#include "globalregt.hpp"

#if VERBOSE_CACHE
 #define _VERBOSE_CACHE 1
#else
 #define _VERBOSE_CACHE 0
 //#define _VERBOSE_CACHE ( sys.getNetwork()->getNodeNum() == 0 )
#endif

namespace nanos {
MemController::MemController( WD *wd, unsigned int numCopies ) :
   _initialized( false )
   , _preinitialized( false )
   , _inputDataReady( false )
   , _outputDataReady( false )
   , _memoryAllocated( false )
   , _invalidating( false )
   , _mainWd( false )
   , _wd( wd )
   , _pe( NULL )
   , _provideLock()
   , _providedRegions()
   , _inOps( NULL )
   , _outOps( NULL )
   , _affinityScore( 0 )
   , _maxAffinityScore( 0 )
   , _ownedRegions()
   , _parentRegions()
   , _memCacheCopies( NULL ) {
   if ( numCopies > 0 ) {
      _memCacheCopies = NEW MemCacheCopy[ numCopies ];
   }
}

MemController::~MemController() {
   delete _inOps;
   delete _outOps;
   delete[] _memCacheCopies;
}

bool MemController::ownsRegion( global_reg_t const &reg ) {
   bool i_has_it = _ownedRegions.hasObjectOfRegion( reg );
   bool parent_has_it  = _parentRegions.hasObjectOfRegion( reg );
   //std::cerr << " wd: " << _wd->getId() << " i has it? " << (i_has_it ? "yes" : "no") << " " << &_ownedRegions << ", parent has it? " << (parent_has_it ? "yes" : "no") << " " << &_parentRegions << std::endl;
   return i_has_it || parent_has_it;
}

void MemController::preInit( ) {
   unsigned int index;
   if ( _preinitialized ) return;
   if ( _VERBOSE_CACHE ) {
      *(myThread->_file) << " (preinit)INITIALIZING MEMCONTROLLER for WD " << _wd->getId() << " " << (_wd->getDescription()!=NULL ? _wd->getDescription() : "n/a")  << " NUM COPIES " << _wd->getNumCopies() << std::endl;
   }

   // std::set<reg_key_t> dicts;
   for ( index = 0; index < _wd->getNumCopies(); index += 1 ) {
      new ( &_memCacheCopies[ index ] ) MemCacheCopy( *_wd, index );
   //    dicts.insert( _memCacheCopies[ index ]._reg.key );
   }
             for ( index = 0; index < _wd->getNumCopies(); index += 1 ) {
                _memCacheCopies[ index ]._reg.id = _memCacheCopies[ index ]._reg.key->obtainRegionId( _wd->getCopies()[index], *_wd, index );
                DirectoryEntryData *entry = ( DirectoryEntryData * ) _memCacheCopies[ index ]._reg.key->getRegionData( _memCacheCopies[ index ]._reg.id );
                if ( entry == NULL ) {
                   entry = NEW DirectoryEntryData();
                   _memCacheCopies[ index ]._reg.key->setRegionData( _memCacheCopies[ index ]._reg.id, entry ); //preInit memCacheCopy._reg
                }
             }
   for ( index = 0; index < _wd->getNumCopies(); index += 1 ) {
      uint64_t host_copy_addr = 0;
      if ( _wd->getParent() != NULL /* && !_wd->getParent()->_mcontrol._mainWd */ ) {
         for ( unsigned int parent_idx = 0; parent_idx < _wd->getParent()->getNumCopies(); parent_idx += 1 ) {
            if ( _wd->getParent()->_mcontrol.getAddress( parent_idx ) == (uint64_t) _wd->getCopies()[ index ].getBaseAddress() ) {
               host_copy_addr = (uint64_t) _wd->getParent()->getCopies()[ parent_idx ].getHostBaseAddress();
               _wd->getCopies()[ index ].setHostBaseAddress( host_copy_addr );
            }
         }
      }
   }

   //std::ostream &o = (*myThread->_file);
   //o << "### preInit wd " << _wd->getId() << std::endl;
   for ( index = 0; index < _wd->getNumCopies(); index += 1 ) {
      //std::cerr << "WD "<< _wd->getId() << " Depth: "<< _wd->getDepth() <<" Creating copy "<< index << std::endl;
      //std::cerr << _wd->getCopies()[ index ];
      //
      //
      //

      // o << "## " << (_wd->getCopies()[index].isInput() ? "in" : "") << (_wd->getCopies()[index].isOutput() ? "out" : "") << " " <<  _wd->getCopies()[index] << std::endl;

      if ( sys.usePredecessorCopyInfo() ) {
         unsigned int predecessorsVersion;
         if ( _providedRegions.hasVersionInfoForRegion( _memCacheCopies[ index ]._reg, predecessorsVersion, _memCacheCopies[ index ]._locations ) ) {
            _memCacheCopies[ index ].setVersion( predecessorsVersion );
         }
      }
      if ( _memCacheCopies[ index ].getVersion() != 0 ) {
         if ( _VERBOSE_CACHE ) { *(myThread->_file) << "WD " << _wd->getId() << " " <<(_wd->getDescription()!=NULL ? _wd->getDescription() : "n/a") << " copy "<< index <<" got location info from predecessor, reg [ "<< (void*)_memCacheCopies[ index ]._reg.key << ","<< _memCacheCopies[ index ]._reg.id << " ] got version " << _memCacheCopies[ index ].getVersion()<< " "; }
         _memCacheCopies[ index ]._locationDataReady = true;
      } else {
         _memCacheCopies[ index ].getVersionInfo();
         if ( _VERBOSE_CACHE ) { *(myThread->_file) << "WD " << _wd->getId() << " " <<(_wd->getDescription()!=NULL ? _wd->getDescription() : "n/a") << " copy "<< index <<" got requesting location info to global directory for region [ "<< (void*)_memCacheCopies[ index ]._reg.key << ","<<  _memCacheCopies[ index ]._reg.id << " ] "; }
      }
      if ( _VERBOSE_CACHE ) {
         for ( NewLocationInfoList::const_iterator it = _memCacheCopies[ index ]._locations.begin(); it != _memCacheCopies[ index ]._locations.end(); it++ ) {
               DirectoryEntryData *rsentry = ( DirectoryEntryData * ) _memCacheCopies[ index ]._reg.key->getRegionData( it->first );
               DirectoryEntryData *dsentry = ( DirectoryEntryData * ) _memCacheCopies[ index ]._reg.key->getRegionData( it->second );
            *(myThread->_file) << "<" << it->first << ": [" << *rsentry << "] ," << it->second << " : [" << *dsentry << "] > ";
         }
         *(myThread->_file) << std::endl;
      }

      if ( _wd->getParent() != NULL && _wd->getParent()->_mcontrol.ownsRegion( _memCacheCopies[ index ]._reg ) ) {
         /* do nothing, maybe here we can add a correctness check,
          * to ensure that the region is a subset of the Parent regions
          */
         //std::cerr << "I am " << _wd->getId() << " parent: " <<  _wd->getParent()->getId() << " NOT ADDING THIS OBJECT "; _memCacheCopies[ index ]._reg.key->printRegion(std::cerr, 1); std::cerr << " adding it to " << &_parentRegions << std::endl;
         _parentRegions.addRegion( _memCacheCopies[ index ]._reg, _memCacheCopies[ index ].getVersion() );
      } else { /* this should be for private data */
         if ( _wd->getParent() != NULL ) {
         //std::cerr << "I am " << _wd->getId() << " parent: " << _wd->getParent()->getId() << " ++++++ ADDING THIS OBJECT "; _memCacheCopies[ index ]._reg.key->printRegion(std::cerr, 1); std::cerr << std::endl;
            _wd->getParent()->_mcontrol._ownedRegions.addRegion( _memCacheCopies[ index ]._reg, _memCacheCopies[ index ].getVersion() );
         }
      }
   }



   //  if ( _VERBOSE_CACHE ) {
   //     //std::ostream &o = (*myThread->_file);
   //     o << "### preInit wd " << _wd->getId() << std::endl;
   //     for ( index = 0; index < _wd->getNumCopies(); index += 1 ) {
   //        o << "## " << (_wd->getCopies()[index].isInput() ? "in" : "") << (_wd->getCopies()[index].isOutput() ? "out" : "") << " "; _memCacheCopies[ index ]._reg.key->printRegion( o, _memCacheCopies[ index ]._reg.id ) ;
   //        o << std::endl;
   //     }
   //  }

      for ( index = 0; index < _wd->getNumCopies(); index += 1 ) {
         std::list< std::pair< reg_t, reg_t > > &missingParts = _memCacheCopies[index]._locations;
         reg_key_t dict = _memCacheCopies[index]._reg.key;
         for ( std::list< std::pair< reg_t, reg_t > >::iterator it = missingParts.begin(); it != missingParts.end(); it++ ) {
            if ( it->first != it->second ) {
               DirectoryEntryData *firstEntry = ( DirectoryEntryData * ) dict->getRegionData( it->first );
               DirectoryEntryData *secondEntry = ( DirectoryEntryData * ) dict->getRegionData( it->second );
               if ( firstEntry == NULL ) {
                  if ( secondEntry != NULL ) {
                     firstEntry = NEW DirectoryEntryData();
                     *firstEntry = *secondEntry;
                  } else {
                     firstEntry = NEW DirectoryEntryData();
                     secondEntry = NEW DirectoryEntryData();
                     dict->setRegionData( it->second, secondEntry ); // preInit fragment
                  }
                  dict->setRegionData( it->first, firstEntry ); //preInit fragment
               } else {
                  if ( secondEntry != NULL ) {
                     *firstEntry = *secondEntry;
                  } else {
                     *myThread->_file << "Dunno what to do..."<<std::endl;
                  }
               }
            } else {
               DirectoryEntryData *entry = ( DirectoryEntryData * ) dict->getRegionData( it->first );
               if ( entry == NULL ) {
                  entry = NEW DirectoryEntryData();
                  dict->setRegionData( it->first, entry ); //preInit fragment
               } else {
               }
            }
         }
      }


   memory_space_id_t rooted_loc = 0;
   if ( this->isRooted( rooted_loc ) ) {
      _wd->tieToLocation( rooted_loc );
   }

   if ( _VERBOSE_CACHE ) {
      *(myThread->_file) << " (preinit)END OF INITIALIZING MEMCONTROLLER for WD " << _wd->getId() << " " << (_wd->getDescription()!=NULL ? _wd->getDescription() : "n/a")  << " NUM COPIES " << _wd->getNumCopies() << " &_preinitialized= "<< &_preinitialized<< std::endl;
   }
   _preinitialized = true;
}

void MemController::initialize( ProcessingElement &pe ) {
   ensure( _preinitialized == true, "MemController not preinitialized!");
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
   ensure( _preinitialized == true, "MemController not preinitialized!");
   ensure( _initialized == true, "MemController not initialized!");
   //std::ostream &o = (*myThread->_file);
   if ( _pe->getMemorySpaceId() != 0 ) {
      bool pending_invalidation = false;
      bool initially_allocated = _memoryAllocated;

      if ( !sys.useFineAllocLock() ) {
      sys.allocLock();
      }

      if ( !_memoryAllocated && !_invalidating ) {
         //o << "### Allocating data for task " << std::dec << _wd->getId() <<  " (" << (_wd->getDescription()!=NULL?_wd->getDescription():"[no desc]")<< ") running on " << std::dec << _pe->getMemorySpaceId() << std::endl;
         bool tmp_result = sys.getSeparateMemory( _pe->getMemorySpaceId() ).prepareRegions( _memCacheCopies, _wd->getNumCopies(), *_wd );
         if ( tmp_result ) {
            for ( unsigned int idx = 0; idx < _wd->getNumCopies() && !pending_invalidation; idx += 1 ) {
               pending_invalidation = (_memCacheCopies[idx]._invalControl._invalOps != NULL);
            }
            if ( pending_invalidation ) {
               _invalidating = true;
               result = false;
            } else {
               _memoryAllocated = true;
            }
         } else {
            result = false;
         }
//if ( tmp_result)         o << "# result:" << ( tmp_result ? (_invalidating ? " invalidating " : " success " ) : " failed " ) << " task " << std::dec << _wd->getId() <<  " (" << (_wd->getDescription()!=NULL?_wd->getDescription():"[no desc]")<< ") running on " << std::dec << _pe->getMemorySpaceId() << std::endl;

      } else if ( _invalidating ) {
         //o << "# process invalidation for wd " << std::dec << _wd->getId() <<  " (" << (_wd->getDescription()!=NULL?_wd->getDescription():"[no desc]")<< ") running on " << std::dec << _pe->getMemorySpaceId() << std::endl;
         for ( unsigned int idx = 0; idx < _wd->getNumCopies(); idx += 1 ) {
            if ( _memCacheCopies[idx]._invalControl._invalOps != NULL ) {
               _memCacheCopies[idx]._invalControl.waitOps( _pe->getMemorySpaceId(), *_wd );

               if ( _memCacheCopies[idx]._invalControl._invalChunk != NULL ) {
                  _memCacheCopies[idx]._chunk = _memCacheCopies[idx]._invalControl._invalChunk;
                  //*(myThread->_file) << "setting invalChunkPtr ( " << _memCacheCopies[idx]._invalControl._invalChunkPtr << " ) <- " << _memCacheCopies[idx]._invalControl._invalChunk << std::endl;
                  *(_memCacheCopies[idx]._invalControl._invalChunkPtr) = _memCacheCopies[idx]._invalControl._invalChunk;
               }

               _memCacheCopies[idx]._invalControl.abort( *_wd );
               delete _memCacheCopies[idx]._invalControl._invalOps;
               _memCacheCopies[idx]._invalControl._invalOps = NULL;
            }
         }
         _invalidating = false;

         bool tmp_result = sys.getSeparateMemory( _pe->getMemorySpaceId() ).prepareRegions( _memCacheCopies, _wd->getNumCopies(), *_wd );
         //o << "# retry result:" << ( tmp_result ? (_invalidating ? " invalidating " : " success " ) : " failed " ) << " task " << std::dec << _wd->getId() <<  " (" << (_wd->getDescription()!=NULL?_wd->getDescription():"[no desc]")<< ") running on " << std::dec << _pe->getMemorySpaceId() << std::endl;
         if ( tmp_result ) {
            pending_invalidation = false;
            for ( unsigned int idx = 0; idx < _wd->getNumCopies() && !pending_invalidation; idx += 1 ) {
               pending_invalidation = (_memCacheCopies[idx]._invalControl._invalOps != NULL);
            }
            if ( pending_invalidation ) {
               _invalidating = true;
               result = false;
            } else {
               _memoryAllocated = true;
            }
         } else {
            result = false;
         }
      } else {
         result = true;
      }

      if ( !initially_allocated && _memoryAllocated ) {
         for ( unsigned int idx = 0; idx < _wd->getNumCopies(); idx += 1 ) {
            int targetChunk = _memCacheCopies[ idx ]._allocFrom;
            if ( targetChunk != -1 ) {
               _memCacheCopies[ idx ]._chunk = _memCacheCopies[ targetChunk ]._chunk;
               // if ( _memCacheCopies[ idx ]._chunk->locked() )
               //    _memCacheCopies[ idx ]._chunk->unlock();
               _memCacheCopies[ idx ]._chunk->addReference( *_wd, 133 ); //allocateTaskMemory, chunk allocated by other copy
            }
         }
      }
      if ( !sys.useFineAllocLock() ) {
      sys.allocUnlock();
      }
   } else {

      _memoryAllocated = true;

      // *(myThread->_file) << "++++ Succeeded allocation for wd " << _wd->getId();
      for ( unsigned int idx = 0; idx < _wd->getNumCopies(); idx += 1 ) {
         // *myThread->_file << " [c: " << (void *) _memCacheCopies[idx]._chunk << " w/hAddr " << (void *) _memCacheCopies[idx]._chunk->getHostAddress() << " - " << (void*)(_memCacheCopies[idx]._chunk->getHostAddress() + _memCacheCopies[idx]._chunk->getSize()) << "]";
         if ( _memCacheCopies[idx]._reg.key->getKeepAtOrigin() ) {
            //std::cerr << "WD " << _wd->getId() << " rooting to memory space " << _pe->getMemorySpaceId() << std::endl;
            _memCacheCopies[idx]._reg.setOwnedMemory( _pe->getMemorySpaceId() );
         }
      }
      //*(myThread->_file)  << std::endl;
   }
   return result;
}

void MemController::copyDataIn() {
   ensure( _preinitialized == true, "MemController not preinitialized!");
   ensure( _initialized == true, "MemController not initialized!");

   if ( _VERBOSE_CACHE || sys.getVerboseCopies() ) {
      //if ( sys.getNetwork()->getNodeNum() == 0 ) {
         std::ostream &o = (*myThread->_file);
         o << "### copyDataIn wd " << std::dec << _wd->getId() << " (" << (_wd->getDescription()!=NULL?_wd->getDescription():"[no desc]")<< ") numCopies "<< _wd->getNumCopies() << " running on " << std::dec << _pe->getMemorySpaceId() << " ops: "<< (void *) _inOps << std::endl;
         for ( unsigned int index = 0; index < _wd->getNumCopies(); index += 1 ) {
         DirectoryEntryData *d = RegionDirectory::getDirectoryEntry( *(_memCacheCopies[ index ]._reg.key), _memCacheCopies[ index ]._reg.id );
         o << "## " << (_wd->getCopies()[index].isInput() ? "in" : "") << (_wd->getCopies()[index].isOutput() ? "out" : "") << " "; _memCacheCopies[ index ]._reg.key->printRegion( o, _memCacheCopies[ index ]._reg.id ) ;
         if ( d ) o << " " << *d << std::endl;
         else o << " dir entry n/a" << std::endl;
         _memCacheCopies[ index ].printLocations( o );
         }
      //}
   }


      sys.allocLock();
   //if( sys.getNetwork()->getNodeNum()== 0)std::cerr << "MemController::copyDataIn for wd " << _wd->getId() << std::endl;
   for ( unsigned int index = 0; index < _wd->getNumCopies(); index++ ) {
      _memCacheCopies[ index ].generateInOps( *_inOps, _wd->getCopies()[index].isInput(), _wd->getCopies()[index].isOutput(), *_wd, index );
   }
      sys.allocUnlock();

   _inOps->issue( _wd );
   if ( _VERBOSE_CACHE || sys.getVerboseCopies() ) {
      if ( sys.getNetwork()->getNodeNum() == 0 ) {
         (*myThread->_file) << "### copyDataIn wd " << std::dec << _wd->getId() << " done" << std::endl;
      }
   }
   //NANOS_INSTRUMENT( inst2.close(); );
}

void MemController::copyDataOut( MemControllerPolicy policy ) {
   ensure( _preinitialized == true, "MemController not preinitialized!");
   ensure( _initialized == true, "MemController not initialized!");

   //for ( unsigned int index = 0; index < _wd->getNumCopies(); index++ ) {
   //   if ( _wd->getCopies()[index].isInput() && _wd->getCopies()[index].isOutput() ) {
   //      _memCacheCopies[ index ]._reg.setLocationAndVersion( _pe->getMemorySpaceId(), _memCacheCopies[ index ].getVersion() + 1 );
   //   }
   //}
   if ( _VERBOSE_CACHE || sys.getVerboseCopies() ) { *(myThread->_file) << "### copyDataOut wd " << std::dec << _wd->getId() << " metadata set, not released yet" << std::endl; }

   for ( unsigned int index = 0; index < _wd->getNumCopies(); index++ ) {
      if ( _wd->getCopies()[index].isOutput() ) {
         if ( _wd->getParent() != NULL && _wd->getParent()->_mcontrol.ownsRegion( _memCacheCopies[index]._reg ) ) {
            WD &parent = *(_wd->getParent());
            for ( unsigned int parent_idx = 0; parent_idx < parent.getNumCopies(); parent_idx += 1) {
            if ( parent._mcontrol._memCacheCopies[parent_idx]._reg.id == 0 ) {
               std::cerr << "Error reg == 0!! 1 parent!"<< std::endl;
            }
            if ( _memCacheCopies[index]._reg.id == 0 ) {
               std::cerr << "Error reg == 0!! 2 this!"<< std::endl;
            }
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

      for ( unsigned int index = 0; index < _wd->getNumCopies(); index++ ) {
         _memCacheCopies[ index ].generateOutOps( &sys.getSeparateMemory( _pe->getMemorySpaceId() ), *_outOps, _wd->getCopies()[index].isInput(), _wd->getCopies()[index].isOutput(), *_wd, index );
      }

      //if( sys.getNetwork()->getNodeNum()== 0)std::cerr << "MemController::copyDataOut for wd " << _wd->getId() << std::endl;
      _outOps->issue( _wd );
   }
}

uint64_t MemController::getAddress( unsigned int index ) const {
   ensure( _preinitialized == true, "MemController not preinitialized!");
   ensure( _initialized == true, "MemController not initialized!");
   uint64_t addr = 0;
   //std::cerr << " _getAddress, reg: " << index << " key: " << (void *)_memCacheCopies[ index ]._reg.key << " id: " << _memCacheCopies[ index ]._reg.id << std::endl;
   if ( _pe->getMemorySpaceId() == 0 ) {
      addr = ((uint64_t) _wd->getCopies()[ index ].getBaseAddress());
   } else {
      addr = sys.getSeparateMemory( _pe->getMemorySpaceId() ).getDeviceAddress( _memCacheCopies[ index ]._reg, (uint64_t) _wd->getCopies()[ index ].getBaseAddress(), _memCacheCopies[ index ]._chunk );
      //std::cerr << "getDevAddr: HostBaseAddr: " << (void*) _wd->getCopies()[ index ].getHostBaseAddress() << " BaseAddr: " << (void*)_wd->getCopies()[ index ].getBaseAddress() << std::endl;
      //std::cerr << "getDevAddr: chunk->getAddress(): " << (void*) _memCacheCopies[ index ]._chunk->getAddress() <<
      //   " - " << (void *) (_memCacheCopies[ index ]._chunk->getAddress() +  _memCacheCopies[ index ]._chunk->getSize()) <<
      //   " chunk size "<< _memCacheCopies[ index ]._chunk->getSize() <<
      //   " chunk->getHostAddress(): " << (void*)_memCacheCopies[ index ]._chunk->getHostAddress() <<
      //   " baseAddress " << (void *) _wd->getCopies()[ index ].getBaseAddress() <<
      //   " offset from _chunk->getHostAddr() - baseAddr: " << ( _memCacheCopies[ index ]._chunk->getHostAddress() - (uint64_t)_wd->getCopies()[ index ].getBaseAddress()) << std::endl;
      //if ( _wd->getCopies()[ index ].isRemoteHost() || _wd->getCopies()[ index ].getHostBaseAddress() == 0 ) {
      //   std::cerr << "Hola" << std::endl;
      //   addr = sys.getSeparateMemory( _pe->getMemorySpaceId() ).getDeviceAddress( _memCacheCopies[ index ]._reg, (uint64_t) _wd->getCopies()[ index ].getBaseAddress(), _memCacheCopies[ index ]._chunk );
      //} else {
      //   std::cerr << "Hola1" << std::endl;
      //   addr = sys.getSeparateMemory( _pe->getMemorySpaceId() ).getDeviceAddress( _memCacheCopies[ index ]._reg, (uint64_t) _wd->getCopies()[ index ].getHostBaseAddress(), _memCacheCopies[ index ]._chunk );
      //}
   }
   //std::cerr << "MemController::getAddress " << (_wd->getDescription()!=NULL?_wd->getDescription():"[no desc]") <<" index: " << index << " addr: " << (void *)addr << std::endl;
   return addr;
}

void MemController::getInfoFromPredecessor( MemController const &predecessorController ) {
   if ( sys.usePredecessorCopyInfo() ) {
      for( unsigned int index = 0; index < predecessorController._wd->getNumCopies(); index += 1) {
         unsigned int version = predecessorController._memCacheCopies[ index ].getChildrenProducedVersion();
         unsigned int predecessorProducedVersion = predecessorController._memCacheCopies[ index ].getVersion() + (predecessorController._wd->getCopies()[ index ].isOutput() ? 1 : 0);
         if ( predecessorProducedVersion == version ) {
            // if the predecessor's children produced new data, then the father can not
            // guarantee that the version is correct (the children may have produced a subchunk
            // of the region). The version is not added here and then the global directory is checked.
            //(*myThread->_file) << "getInfoFromPredecessor[ " << _wd->getId() << " : "<< _wd->getDescription()<< " key: " << (void*)predecessorController._memCacheCopies[ index ]._reg.key << " ] adding version " << version << " from wd " << predecessorController._wd->getId() << " : " << predecessorController._wd->getDescription() << " : " << index << " copy version " << version << std::endl;
            _providedRegions.addRegion( predecessorController._memCacheCopies[ index ]._reg, version );
         } else {
            //(*myThread->_file) << _preinitialized << " " << _initialized <<" SKIP getInfoFromPredecessor[ " << _wd->getId() << " : "<< _wd->getDescription()<< " key: " << (void*)predecessorController._memCacheCopies[ index ]._reg.key << " ] adding version " << version << " from wd " << predecessorController._wd->getId() << " : " << predecessorController._wd->getDescription() << " : " << index << " copy CP version " << version << " predec Produced version " << predecessorProducedVersion << std::endl;
         }
      }
   }
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
   if ( _preinitialized != true ) {
      std::cerr << " CANT CALL " << __func__ << " BEFORE PRE INIT (wd: " << wd.getId() << " - " << (wd.getDescription() != NULL ? wd.getDescription() : "n/a") << ")" << std::endl;
   }
   ensure( _preinitialized == true, "MemController not initialized! wd " );
   if ( _initialized ) {
      if ( !_outputDataReady ) {
         _outputDataReady = _outOps->isDataReady( wd );
         if ( _outputDataReady ) {
            if ( _VERBOSE_CACHE ) { *(myThread->_file) << "Output data is ready for wd " << _wd->getId() << " obj " << (void *)_outOps << std::endl; }

            sys.getSeparateMemory( _pe->getMemorySpaceId() ).releaseRegions( _memCacheCopies, _wd->getNumCopies(), *_wd ) ;
            //for ( unsigned int index = 0; index < _wd->getNumCopies(); index++ ) {
            //   sys.getSeparateMemory( _pe->getMemorySpaceId() ).releaseRegions( _memCacheCopies, _wd->getNumCopies(), *_wd ) ;
            //}
         }
      }
      return _outputDataReady;
   }
   return false;
}

bool MemController::canAllocateMemory( memory_space_id_t memId, bool considerInvalidations ) const {
   if ( memId > 0 ) {
      return sys.getSeparateMemory( memId ).canAllocateMemory( _memCacheCopies, _wd->getNumCopies(), considerInvalidations, *_wd );
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
   for ( unsigned int index = 0; index < _wd->getNumCopies(); index++ ) {
      total += _memCacheCopies[ index ]._reg.getDataSize();
   }
   return total;
}

bool MemController::isRooted( memory_space_id_t &loc ) const {
   bool result = false;
   memory_space_id_t refLoc = (memory_space_id_t) -1;
   for ( unsigned int index = 0; index < _wd->getNumCopies(); index++ ) {
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

bool MemController::isMultipleRooted( std::list<memory_space_id_t> &locs ) const {
   unsigned int count = 0;
   for ( unsigned int index = 0; index < _wd->getNumCopies(); index++ ) {
      memory_space_id_t thisLoc;
      if ( _memCacheCopies[ index ].isRooted( thisLoc ) ) {
         count += 1;
         locs.push_back( thisLoc );
      }
   }
   return count > 1;
}

void MemController::setMainWD() {
   _mainWd = true;
}


void MemController::synchronize( std::size_t numDataAccesses, DataAccess *data ) {
   sys.getHostMemory().synchronize( *_wd, numDataAccesses, data );
}
void MemController::synchronize() {
   sys.getHostMemory().synchronize( *_wd );
}

bool MemController::isMemoryAllocated() const {
   return _memoryAllocated;
}

void MemController::setCacheMetaData() {
   for ( unsigned int index = 0; index < _wd->getNumCopies(); index++ ) {
      if ( _wd->getCopies()[index].isOutput() ) {
         unsigned int newVersion = _memCacheCopies[ index ].getVersion() + 1;
         _memCacheCopies[ index ]._reg.setLocationAndVersion( _pe, _pe->getMemorySpaceId(), newVersion ); //update directory, OUT copies, (upgrade version)
         //*myThread->_file << "setChildrenProducedVersion( " << newVersion << " ) for WD " << _wd->getId() << " : " << _wd->getDescription() << std::endl;
         _memCacheCopies[ index ].setChildrenProducedVersion( newVersion );
         if ( _pe->getMemorySpaceId() != 0 /* HOST_MEMSPACE_ID */) {
            //*myThread->_file <<__func__<< " setRegionVersion ( " << newVersion << " ) for WD " << _wd->getId() << " : " << _wd->getDescription() << " and index " << index <<std::endl;
            sys.getSeparateMemory( _pe->getMemorySpaceId() ).setRegionVersion( _memCacheCopies[ index ]._reg, _memCacheCopies[ index ]._chunk, newVersion, *_wd, index );
         }
      } else if ( _wd->getCopies()[index].isInput() ) {
         _memCacheCopies[ index ].setChildrenProducedVersion( _memCacheCopies[ index ].getVersion() );
      }
   }
}

bool MemController::hasObjectOfRegion( global_reg_t const &reg ) {
   return _ownedRegions.hasObjectOfRegion( reg );
}


bool MemController::containsAllCopies( MemController const &target ) const {
   bool result = true;
   for ( unsigned int idx = 0; idx < target._wd->getNumCopies() && result; idx += 1 ) {
      bool this_reg_is_contained = false;
      for ( unsigned int this_idx = 0; this_idx < _wd->getNumCopies() && !this_reg_is_contained; this_idx += 1 ) {
         //   std::cerr << "this_reg "; target._memCacheCopies[idx]._reg.key->printRegion(std::cerr, target._memCacheCopies[idx]._reg.id); std::cerr << std::endl;
         //   std::cerr << "target_reg "; _memCacheCopies[this_idx]._reg.key->printRegion(std::cerr, _memCacheCopies[this_idx]._reg.id); std::cerr << std::endl;
         if ( target._memCacheCopies[idx]._reg.key == _memCacheCopies[this_idx]._reg.key ) {
            reg_key_t key = target._memCacheCopies[idx]._reg.key;
            reg_t this_reg = _memCacheCopies[this_idx]._reg.id;
            reg_t target_reg = target._memCacheCopies[idx]._reg.id;
            if ( target_reg == this_reg || ( key->checkIntersect( target_reg, this_reg ) && key->computeIntersect( target_reg, this_reg ) == target_reg ) ) {
               this_reg_is_contained = true;
            }
         //   std::cerr << "same key [target idx = " << idx << "]= "<< target_reg <<" && [this idx = " << this_idx << "]= " << this_reg << " result " << this_reg_is_contained << std::endl;
         //   std::cerr << "this_reg "; key->printRegion(std::cerr, this_reg); std::cerr << std::endl;
         //   std::cerr << "target_reg "; key->printRegion(std::cerr, target_reg); std::cerr << std::endl;
         //} else {
         //   std::cerr << "diff key [target idx = " << idx << "] && [this idx = " << this_idx << "] result " << this_reg_is_contained << std::endl;
         }
      }
      result = this_reg_is_contained;
   }
   return result;
}

}
