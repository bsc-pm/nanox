#include "system.hpp"
#include "network_decl.hpp"
#include "remoteworkgroup_decl.hpp"

using namespace nanos;

RemoteWorkGroup::RemoteWorkGroup( unsigned int rId ) : WorkGroup(), _remoteId ( rId ) {
}

void RemoteWorkGroup::exitWork( WorkGroup &work ) { 
   sys.getNetwork()->sendWorkDoneMsg( Network::MASTER_NODE_NUM, work.getRemoteAddr() , _remoteId );
}
