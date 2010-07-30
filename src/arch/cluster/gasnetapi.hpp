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
         void sendWorkMsg ( unsigned int dest, void ( *work ) ( void * ), unsigned int arg0, size_t argSize, void * arg );
         void sendWorkDoneMsg ( unsigned int dest );
         void put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, size_t size );
         void get ( void *localAddr, unsigned int remoteNode, uint64_t remoteAddr, size_t size );
         void malloc ( unsigned int remoteNode, size_t size );
   };
}
}
#endif
