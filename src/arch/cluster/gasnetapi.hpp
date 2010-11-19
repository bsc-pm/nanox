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


#ifndef _GASNET_API
#define _GASNET_API

#include "network.hpp"
#include "networkapi.hpp"

namespace nanos {
namespace ext {

   class GasnetAPI : public NetworkAPI
   {
      private:
         Network *_net;
         
      public:
         void initialize ( Network *net );
         void finalize ();
         void poll ();
         void sendExitMsg ( unsigned int dest );
         void sendWorkMsg ( unsigned int dest, void ( *work ) ( void * ), unsigned int arg0, unsigned int arg1, unsigned int numPe, size_t argSize, void * arg );
         void sendWorkDoneMsg ( unsigned int dest, unsigned int numPe );
         void put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, size_t size );
         void get ( void *localAddr, unsigned int remoteNode, uint64_t remoteAddr, size_t size );
         void malloc ( unsigned int remoteNode, size_t size, unsigned int id );
         void sendMyHostName( unsigned int dest );
   };
}
}
#endif
