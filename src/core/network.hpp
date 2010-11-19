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


#ifndef _NANOX_NETWORK
#define _NANOX_NETWORK

#include "networkapi.hpp"

namespace nanos {

   class Network
   {
      private:
         unsigned int _numNodes;
         NetworkAPI *_api; 
         unsigned int _nodeNum;
         volatile unsigned int *_notify;
         void **_malloc_return;
         bool *_malloc_complete;
         char *_masterHostname;

      public:
         static const unsigned int MASTER_NODE_NUM = 0;
         // constructor
         Network ();
         ~Network ();

         void setAPI ( NetworkAPI *api );
         NetworkAPI *getAPI ();
         void setNumNodes ( unsigned int numNodes );
         unsigned int getNumNodes ();
         void setNodeNum ( unsigned int nodeNum );
         unsigned int getNodeNum ();
         void notifyWorkDone ( unsigned int nodeNum, unsigned int numPe );
         void notifyMalloc ( unsigned int nodeNum, void * result, unsigned int id );

         void initialize ( void );
         void finalize ( void );
         void poll ( void );
         void sendExitMsg( unsigned int nodeNum );
         void sendWorkMsg( unsigned int dest, void ( *work ) ( void * ), unsigned int arg0, unsigned int arg1, unsigned int numPe, size_t argSize, void * arg );
         void sendWorkDoneMsg( unsigned int nodeNum, unsigned int numPe );
         void put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, size_t size );
         void get ( void *localAddr, unsigned int remoteNode, uint64_t remoteAddr, size_t size );
         void * malloc ( unsigned int remoteNode, size_t size, unsigned int id );
         void setMasterHostname( char *name );
         char *getMasterHostname();
   };
}

#endif
