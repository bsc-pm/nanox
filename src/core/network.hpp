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
         unsigned int *_notify;
         void **_malloc_return;
         bool *_malloc_complete;

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
         void notifyWorkDone ( unsigned int nodeNum );
         void notifyMalloc ( unsigned int nodeNum, void * result );

         void initialize ( void );
         void poll ( void );
         void sendExitMsg( unsigned int nodeNum );
         void sendWorkMsg( unsigned int dest, void ( *work ) ( void * ), unsigned int arg0, size_t argSize, void * arg );
         void sendWorkDoneMsg( unsigned int nodeNum );
         void put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, size_t size );
         void get ( void *localAddr, unsigned int remoteNode, uint64_t remoteAddr, size_t size );
         void * malloc ( unsigned int remoteNode, size_t size );
   };
}

#endif
