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

#include "basethread_decl.hpp"
#include "invalidationcontroller_decl.hpp"
#include "regioncache.hpp"
#include "system_decl.hpp"
#include "globalregt.hpp"
#include "os.hpp"
#include <limits>
#include <iomanip>

using namespace nanos;

InvalidationController::InvalidationController() : 
   _invalOps( NULL )
   , _invalChunk( NULL )
   , _invalChunkPtr( NULL )
   , _regions_to_remove_access()
   , _chunksToFree()
   , _chunksToInval()
   , _allocatedRegion()
   , _softInvalidationCount( 0 )
   , _hardInvalidationCount( 0 ) {
}

InvalidationController::~InvalidationController() {
   delete _invalOps;
}

void InvalidationController::abort(WD const &wd) {
//sys.getSeparateMemory(id).getCache().lock();
   for ( std::set< std::pair< AllocatedChunk **, AllocatedChunk * > >::iterator it = _chunksToInval.begin(); it != _chunksToInval.end(); it++ ) {
      if ( *it->first == (AllocatedChunk *) -1 ) {
         it->second->removeReference( wd ); //InvalidationController::abort
         *(it->first) = it->second;
      }
   }
   if ( _invalChunkPtr != NULL ) {
      if ( *_invalChunkPtr == (AllocatedChunk *) -2 ) {
         *_invalChunkPtr = (AllocatedChunk *) NULL; 
      }
   }
   _chunksToInval.clear();
   _chunksToFree.clear();
   _regions_to_remove_access.clear();
//sys.getSeparateMemory(id).getCache().unlock();
   if ( _invalOps != NULL ) {
      _invalOps->cancel(wd);
      delete _invalOps;
   }
   new (this) InvalidationController();
}

bool InvalidationController::isInvalidating() const {
   return _invalOps != NULL /*&& _invalOps->hasPendingOps()*/;
}

void InvalidationController::postIssueActions( memory_space_id_t id ) {
   sys.getSeparateMemory(id).getCache().lock();
   sys.getSeparateMemory(id).getCache().addToAllocatedRegionMap( _allocatedRegion );
   sys.getSeparateMemory(id).getCache().MAPlock();
   for ( std::set< std::pair< AllocatedChunk **, AllocatedChunk * > >::iterator it = _chunksToInval.begin(); it != _chunksToInval.end(); it++ ) {
      sys.getSeparateMemory(id).getCache().removeFromAllocatedRegionMap( it->second->getAllocatedRegion() );
   }
   sys.getSeparateMemory(id).getCache().MAPunlock();
   sys.getSeparateMemory(id).getCache().unlock();
}

void InvalidationController::waitOps( memory_space_id_t id, WD const &wd ) {
   //*myThread->_file << std::setprecision(std::numeric_limits<double>::digits10) << OS::getMonotonicTime() << " invalidation for wd " << wd.getId() << " wait for ownOps: " << _invalOps->getOwnOps().size() << " otherOps: " << _invalOps->getOtherOps().size() << std::endl;
   //_invalOps->print(*myThread->_file);
   while ( !_invalOps->isDataReady( wd, true ) ) { myThread->processTransfers(); }
   //*myThread->_file << std::setprecision(std::numeric_limits<double>::digits10) << OS::getMonotonicTime() << " invalidation for wd " << wd.getId() << " wait done _chunksToFree size= " << _chunksToFree.size() << " and _regions_to_remove_access.size= " << _regions_to_remove_access.size() << std::endl;
   postCompleteActions( id, wd );
}

void InvalidationController::preIssueActions( memory_space_id_t id, WD const &wd ) {
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t key = ID->getEventKey("cache-free"); )
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent( key, (nanos_event_value_t) wd.getId() ); )
   //if ( _VERBOSE_CACHE ) { std::cerr << "===> Invalidation complete at " << _memorySpaceId << " remove access for regs: "; }
   for ( std::set< global_reg_t >::iterator it = _regions_to_remove_access.begin(); it != _regions_to_remove_access.end(); it++ ) {
   //if ( _VERBOSE_CACHE ) { std::cerr << it->id << " "; }
      RegionDirectory::delAccess( it->key, it->id, id );
   }
   //if ( _VERBOSE_CACHE ) { std::cerr << std::endl ; }
}

void InvalidationController::postCompleteActions( memory_space_id_t id, WD const &wd ) {
   if ( _invalChunk ) {
      uint64_t targetHostAddr = _allocatedRegion.getRealFirstAddress();
      _invalChunk->increaseLruStamp();
      _invalChunk->clearNewRegions( _allocatedRegion );
      _invalChunk->setHostAddress( targetHostAddr );
   }
   for ( std::set< AllocatedChunk * >::iterator it = _chunksToFree.begin(); it != _chunksToFree.end(); it++ ) {
      sys.getSeparateMemory(id).getCache().freeChunk( *it, wd );
   }
   sys.getSeparateMemory(id).getCache().lock();
   // if ( _softInvalidationCount > 0 ) {
   //    std::cerr << "soft inval at wd " << wd.getId() << " " << wd.getDescription() << std::endl;
   // }
   sys.getSeparateMemory(id).getCache().increaseSoftInvalidationCount( _softInvalidationCount );
   sys.getSeparateMemory(id).getCache().increaseHardInvalidationCount( _hardInvalidationCount );
   for ( std::set< std::pair< AllocatedChunk **, AllocatedChunk * > >::iterator it = _chunksToInval.begin(); it != _chunksToInval.end(); it++ ) {
      *(it->first) = NULL;
   }
   sys.getSeparateMemory(id).getCache().unlock();
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t key = ID->getEventKey("cache-free"); )
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( key, (nanos_event_value_t) 0 ); )
}
