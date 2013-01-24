#include "system.hpp"
#include "remoteworkgroup_decl.hpp"

using namespace nanos;

RemoteWorkGroup::RemoteWorkGroup(unsigned int rId) : WorkGroup(), _remoteId ( rId ) {
}

void RemoteWorkGroup::exitWork( WorkGroup &work ) { 
   sys.getNetwork()->sendWorkDoneMsg( Network::MASTER_NODE_NUM, /*new queue */work.getRemoteAddr() , _remoteId);
}
