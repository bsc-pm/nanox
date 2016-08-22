/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

#include <stdio.h>

#include "memcachecopy_decl.hpp"
#include "system_decl.hpp"
#include "memoryops_decl.hpp"
#include "deviceops.hpp"
#include "workdescriptor.hpp"
#include "basethread.hpp"
#include "regiondict.hpp"

using namespace nanos;

void MemCacheCopy::generateInOps( BaseAddressSpaceInOps &ops, bool input, bool output, WD const &wd, unsigned int copyIdx ) {
   //NANOS_INSTRUMENT( InstrumentState inst4(NANOS_CC_CDIN_OP_GEN); );
   _reg.key->lockObject(); //FIXME: should this be a red/write lock (in read mode)?
   if ( input && output ) {
      //re read version, in case of this being a commutative or concurrent access
      if ( _reg.getVersion() > _version ) {
         *myThread->_file << "[!!!] WARNING: concurrent or commutative detected, wd " << wd.getId() << " " << (wd.getDescription()!=NULL?wd.getDescription():"[no desc]") << " index " << copyIdx << " _reg.getVersion() " << _reg.getVersion() << " _version " << _version << std::endl;
         _version = _reg.getVersion();
      }
   }

   if ( ops.getPE()->getMemorySpaceId() != 0 ) {
      /* CACHE ACCESS */
      if ( input )  {
         if ( _policy == RegionCache::FPGA ) {
            _chunk->copyRegionFromHost( ops, _reg.id, _version, wd, copyIdx );
         } else {
            _chunk->NEWaddReadRegion2( ops, _reg.id, _version, _locations, wd, copyIdx );
         }
      } else if ( output ) {
         _chunk->NEWaddWriteRegion( _reg.id, _version, &wd, copyIdx );
      } else {
         fatal("invalid path");
      }
   } else {
      /* HOST ACCESS */
      if ( input )  {
         ops.copyInputData( *this, wd, copyIdx );
      }
   }
   //NANOS_INSTRUMENT( inst4.close(); );
   _reg.key->unlockObject(); //FIXME: should this be a red/write lock (in read mode)?

}

void MemCacheCopy::release( memory_space_id_t loc, WD const &wd, unsigned int copyIdx ) {
   int refs = _chunk->removeReference( wd ); //MemCacheCopy::release (new)
   if ( refs == 1 && ( _policy == RegionCache::NO_CACHE || _policy == RegionCache::FPGA ) ) {

      //deallocate
      sys.getSeparateMemory( loc ).getCache().freeChunk( _chunk, false, wd, copyIdx );
      //deallocate end
      RegionDirectory::delAccess( _reg.key, _reg.id, loc );
   }
}

bool MemCacheCopy::allocate( memory_space_id_t loc, WD const &wd, unsigned int copyIdx ) {
   bool result;
   //addReference here? A: NO!! it should be added when we ask for the chunk (getChunk)
   // update, is better to addReference here, if we fail to allocate it may be worth to release the chunks, otherwise we may deadlock
   //update2, we can not addReference here. This function may be called multiple times for the same chunk, which will add many references.
             NANOS_INSTRUMENT(static nanos_event_key_t ikey = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("debug");)
             NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( ikey, 99556 );)
   _chunk->lock_AllocatedChunk();
             NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( ikey, 88556 );)
   if ( !_chunk->allocated() ) {
      if ( _chunk->isInvalidating() ) {
         result = false;
         *myThread->_file << "wait, invalidating" << std::endl;
      } else {
 //        NANOS_INSTRUMENT(static nanos_event_key_t ikey = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("debug");)
         NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( ikey, 777 );)
         result = sys.getSeparateMemory( loc ).getCache().allocateChunk( _chunk, wd, copyIdx );
         NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( ikey, 0 );)
         if ( result == false ) {
         NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( ikey, 888 );)
            result = sys.getSeparateMemory( loc ).getCache().invalidate( *_chunk, _chunk->getAllocatedRegion().getDataSize(), wd );
         NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( ikey, 0 );)
         }
      }
   } else {
      result = true;
   }
             NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( ikey, 0 );)
   _chunk->unlock_AllocatedChunk();
             NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( ikey, 0 );)
   return result;
}
