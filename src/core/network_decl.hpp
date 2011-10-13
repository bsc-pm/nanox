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

#include <string>
#include "networkapi.hpp"

namespace nanos {

   class Network
   {
      private:
         unsigned int _numNodes;
         NetworkAPI *_api; 
         unsigned int _nodeNum;

         //std::string _masterHostname;
         char * _masterHostname;

      public:
         static const unsigned int MASTER_NODE_NUM = 0;
         typedef struct {
            int complete;
            void * resultAddr;
         } mallocWaitObj;
         // constructor

         Network ();
         ~Network ();

         void setAPI ( NetworkAPI *api );
         NetworkAPI *getAPI ();
         void setNumNodes ( unsigned int numNodes );
         unsigned int getNumNodes () const;
         void setNodeNum ( unsigned int nodeNum );
         unsigned int getNodeNum () const;
         void notifyWorkDone ( unsigned int nodeNum, void *remoteWdAddr, int peId );
         void notifyMalloc ( unsigned int nodeNum, void * result, mallocWaitObj *waitObjAddr );

         void initialize ( void );
         void finalize ( void );
         void poll ( unsigned int id );
         void sendExitMsg( unsigned int nodeNum );
         void sendWorkMsg( unsigned int dest, void ( *work ) ( void * ), unsigned int arg0, unsigned int arg1, unsigned int numPe, size_t argSize, char * arg, void ( *xlate ) ( void *, void * ), int arch, void *remoteWdAddr );
         bool isWorking( unsigned int dest, unsigned int numPe ) const;
         void sendWorkDoneMsg( unsigned int nodeNum, void *remoteWdaddr, int peId );
         void put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, size_t size );
         void get ( void *localAddr, unsigned int remoteNode, uint64_t remoteAddr, size_t size );
         void *malloc ( unsigned int remoteNode, size_t size );
         void memFree ( unsigned int remoteNode, void *addr );
         void memRealloc ( unsigned int remoteNode, void *oldAddr, size_t oldSize, void *newAddr, size_t newSize );
         void nodeBarrier( void );

         void setMasterHostname( char *name );
         //const std::string & getMasterHostname( void ) const;
         const char * getMasterHostname( void ) const;
         void sendRequestPut( unsigned int dest, uint64_t origAddr, unsigned int dataDest, uint64_t dstAddr, size_t len );
         void setMasterDirectory(Directory *dir);
         std::size_t getTotalBytes();

         static Lock _nodeLock;
         static Atomic<uint64_t> _nodeRCAaddr;
         static Atomic<uint64_t> _nodeRCAaddrOther;
   };
}

#endif
