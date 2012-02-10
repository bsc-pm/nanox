/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
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
#ifndef _NANOS_MEMTRACKER_HPP
#define _NANOS_MEMTRACKER_HPP

#ifdef NANOS_DEBUG_ENABLED

#include "memtracker_decl.hpp"
#include "new_decl.hpp"
#include "allocator.hpp"
#include "config.hpp"

namespace nanos {

   inline MemTracker::MemTracker() : _blocks(),_stats(), _totalMem( 0 ), _numBlocks( 0 ),_maxMem( 0 ), _lock(), _showStats( false )
   {
      Config config;
      config.registerConfigOption ( "mem-stats", NEW Config::FlagOption( _showStats ), "Show memory leak stats" );
      config.registerArgOption ( "mem-stats", "mem-stats" );
      config.registerEnvOption ( "mem-stats", "NX_DEBUG_MEM_STATS" );
      config.init();
   }

   inline void * MemTracker::allocate ( size_t size, const char *file, int line )
   {
      int thread_id;

      BaseThread *thread = getMyThreadSafe();

      if ( thread != NULL ) thread_id = thread->getId();
      else thread_id = -1;

      LockBlock_noinst guard(_lock);

#ifdef NANOS_DISABLE_ALLOCATOR
      void *p = malloc ( size );
#else
      void *p = nanos::getAllocator().allocate( size );
#endif

      if ( p ) {
         _blocks[p] = BlockInfo(size,file,line, thread_id);
         _numBlocks++;
         _totalMem += size;
         _stats[size]._current++;
         _stats[size]._total++;
         _stats[size]._max = std::max( _stats[size]._max, _stats[size]._current );
         _maxMem = std::max( _maxMem, _totalMem );
      } else {
         throw std::bad_alloc();
      }

      return p;
   }

   inline void MemTracker::deallocate ( void * p, const char *file, int line )
   {
      LockBlock_noinst guard(_lock);

      AddrMap::iterator it = _blocks.find( p );

      if ( it != _blocks.end() ) {
         _numBlocks--;
         _totalMem -= it->second._size;

#ifdef NANOS_DISABLE_ALLOCATOR
         free( p );
#else
         nanos::getAllocator().deallocate( p );
#endif

         _stats[it->second._size]._current--;
         _blocks.erase( it );
      } else {
         guard.release();

         if ( file != NULL ) {
            message0("Trying to free invalid pointer " << p << " at " << file << ":" << line);
         } else {
            message0("Trying to free invalid pointer " << p);
         }    
         throw std::bad_alloc();
      }
   }

   inline void MemTracker::showStats ()
   {
      if ( !_showStats ) return;

      message0("======================= General Memory stats ============");
      message0("# blocks              " << _numBlocks);
      message0("total unfreed memory  " << _totalMem << " bytes");
      message0("max allocated memory  " << _maxMem << " bytes");
      message0("=========================================================");
      message0("======================= Unfreed blocks ==================");
      for ( AddrMap::iterator it = _blocks.begin(); it != _blocks.end(); it++ )
      {
         BlockInfo &info = it->second;
         if ( info._file != NULL ) {
            message0(info._size << " bytes allocated in " << info._file << ":" << info._line << " @ thread " << info._thread );
         } else {
            message0(info._size << " bytes allocated in an unknown location @ thread " << info._thread );
         }
      }
      message0("=========================================================");
   }

} // namespace nanos

#endif
#endif
