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

#ifndef SHAREDMEMALLOCATOR_DECL_HPP
#define	SHAREDMEMALLOCATOR_DECL_HPP

#include <stdint.h>
#include <map>
#include "openclprocessor_fwd.hpp"

#include "lock.hpp"

namespace nanos {
    
/*! \brief Memory allocator to manage shared memory memory allocations
    */
   class SharedMemAllocator
   {
      private:
         typedef std::map < void *, size_t > SharedMemMemoryMap;

         SharedMemMemoryMap            _pinnedChunks;
         nanos::ext::OpenCLProcessor* _allocatingDevice;
         bool _init;
         Lock                       _lock;
         
         void initialize();


      public:
         SharedMemAllocator();
        ~SharedMemAllocator () { }

         void * allocate( size_t size );

         void free( void * address );

         bool isSharedMem( void * address, size_t size );
         
         void* getBasePointer( void * address, size_t size );

         void printSharedMemMemoryMap();
         
         bool initialized(){
             return _init;
         }
   };

} // namespace nanos
#endif	/* SHAREDMEMALLOCATOR_DECL_HPP */

