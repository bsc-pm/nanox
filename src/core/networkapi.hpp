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
         virtual void sendWorkMsg ( unsigned int dest, void ( *work ) ( void * ), unsigned int arg0, size_t argSize, void * arg ) = 0;
         virtual void sendWorkDoneMsg ( unsigned int dest ) = 0;
         virtual void put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, size_t size ) = 0;
         virtual void get ( void *localAddr, unsigned int remoteNode, uint64_t remoteAddr, size_t size ) = 0;
         virtual void malloc ( unsigned int remoteNode, size_t size ) = 0;
   };
}

#endif
