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

namespace nanos {

   class Network;

   class NetworkAPI
   {
      public:
         virtual void initialize ( Network *net ) = 0;
         virtual void finalize () = 0;
         virtual void poll () = 0;
         virtual void sendExitMsg ( unsigned int dest ) = 0;
         virtual void sendWorkMsg ( unsigned int dest, void ( *work ) ( void * ), unsigned int arg0, unsigned int arg1, unsigned int numPe, size_t argSize, void * arg ) = 0;
         virtual void sendWorkDoneMsg ( unsigned int dest, unsigned int numPe ) = 0;
         virtual void put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, size_t size ) = 0;
         virtual void get ( void *localAddr, unsigned int remoteNode, uint64_t remoteAddr, size_t size ) = 0;
         virtual void malloc ( unsigned int remoteNode, size_t size, unsigned int id ) = 0;
         virtual void nodeBarrier( void ) = 0;
         virtual void getNotify ( unsigned int node, uint64_t remoteAddr ) = 0;
   };
}

#endif
