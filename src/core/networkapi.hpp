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

#ifndef _NANOX_NETWORK_API
#define _NANOX_NETWORK_API

#include <stdint.h>
#include <vector>
#include "directory.hpp"

namespace nanos {

   class Network;

   class NetworkAPI
   {
      public:
         virtual void initialize ( Network *net ) = 0;
         virtual void finalize () = 0;
         virtual void poll () = 0;
         virtual void sendExitMsg ( unsigned int dest ) = 0;
         virtual void sendWorkMsg ( unsigned int dest, void ( *work ) ( void * ), unsigned int arg0, unsigned int arg1, unsigned int numPe, size_t argSize, char * arg, void ( *xlate ) ( void *, void * ), int arch , void *remoteWdAddr) = 0;
         virtual void sendWorkDoneMsg ( unsigned int dest, void *remoteWdAddr, int peId ) = 0;
         virtual void put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, size_t size ) = 0;
         virtual void get ( void *localAddr, unsigned int remoteNode, uint64_t remoteAddr, size_t size ) = 0;
         virtual void malloc ( unsigned int remoteNode, size_t size, void * waitObjAddr) = 0;
         virtual void memFree ( unsigned int remoteNode, void *addr ) = 0;
         virtual void memRealloc ( unsigned int remoteNode, void *oldAddr, size_t oldSize, void *newAddr, size_t newSize ) = 0;
         virtual void nodeBarrier( void ) = 0;
         //virtual void getNotify ( unsigned int node, uint64_t remoteAddr ) = 0;
         virtual void sendRequestPut( unsigned int dest, uint64_t origAddr, unsigned int dataDest, uint64_t dstAddr, size_t len ) = 0;
         virtual std::size_t getTotalBytes() = 0;
        
         virtual void setMasterDirectory(Directory *d) = 0;
         //virtual void setGpuCache(Cache *_cache) = 0;
   };
}

#endif
