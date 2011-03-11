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
#include "memtracker_decl.hpp"
#include "allocator.hpp"

namespace nanos {

      inline void * MemTracker::allocate ( size_t size, const char *file, int line )
      {
         LockBlock guard(_lock);

//         void *p = malloc( size );
           void *p = nanos::getAllocator().allocate( size );


	if ( p ) {
	    _blocks[p] = BlockInfo(size,file,line);
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
	LockBlock guard(_lock);
	
	AddrMap::iterator it = _blocks.find( p );

	if ( it != _blocks.end() ) {
	    _numBlocks--;
	    _totalMem -= it->second._size;

	    //free( p );
            nanos::getAllocator().deallocate( p );

	    _blocks.erase( it );
	    _stats[it->second._size]._current--;
	} else {
	    guard.release();
	    
	    if ( file != NULL ) {
	      message0("Trying to free invalid pointer " << p << " at " << file << ":" << line);
	    } else {
	      message0("Trying to free invalid pointer " << p);
	    }    
	}
      }

      inline void MemTracker::showStats ()
      {
	message0("======================= General Memory stats ============");
	std::cout
	    << "# blocks              " << _numBlocks << std::endl
	    << "total unfreed memory  " << _totalMem << " bytes" << std::endl
	    << "max allocated memory  " << _maxMem << " bytes" << std::endl
	    ;
        message0("=========================================================");
	message0("======================= Unfreed blocks ==================");
	for ( AddrMap::iterator it = _blocks.begin(); it != _blocks.end(); it++ )
	{
	    BlockInfo &info = it->second;
	    if ( info._file != NULL ) {
	      message0(info._size << " bytes allocated in " << info._file << ":" << info._line);
	    } else {
	      message0(info._size << " bytes allocated in an unknown location");
	    }
	}
        message0("=========================================================");
#if 0        
	message0("======================= Block Sizes Stats ===============");
	message0("Size   Unfreed   Max   Total");
	for ( SizeMap::iterator it = _stats.begin(); it != _stats.end(); it ++ ) {
	    DistrInfo &info = it->second;
	    message0(it->first << " " << info._current << " " << info._max << " " << info._total );    
	}
	message0("=========================================================");
#endif        
      }

} // namespace nanos

#endif
