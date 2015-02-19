#include "system.hpp"
#include "network_decl.hpp"
#include "remoteworkdescriptor_decl.hpp"

using namespace nanos;

RemoteWorkDescriptor::RemoteWorkDescriptor( unsigned int rId ) : WorkDescriptor( 0, NULL ), _remoteId ( rId ) {
   if ( sys.getPMInterface().getInternalDataSize() > 0 ) {
      char *data = NEW char[sys.getPMInterface().getInternalDataSize()];
      sys.getPMInterface().initInternalData( data );
      this->setInternalData( data );
   }
}

void RemoteWorkDescriptor::exitWork( WorkDescriptor &work ) { 
   WorkDescriptor::exitWork( work );
   sys.getNetwork()->sendWorkDoneMsg( Network::MASTER_NODE_NUM, work.getRemoteAddr() , _remoteId );
}
