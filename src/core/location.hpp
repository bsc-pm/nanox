#ifndef LOCATION_H
#define LOCATION_H

#include "location_decl.hpp"

using namespace nanos;

//inline Location::Location() : _nodeId( (unsigned int) -1 ), _memorySpaceId( ( unsigned int ) -1 ), _socketId( ( unsigned int ) -1 ), _coreId( (unsigned int) -1 ) { }
inline Location::Location( unsigned int clusterNode, unsigned int numaNode, bool inNumaNode, unsigned int socket, bool inSocket) :
      _clusterNode( clusterNode ), _numaNode( numaNode ), _socket( socket ), _inNumaNode( inNumaNode ), _inSocket( inSocket ) { }
/*
inline Location::Location( Location const &l) : _nodeId( l._nodeId ), _memorySpaceId( l._memorySpaceId ), _socketId( l._socketId ), _coreId( l._coreId ) { }
inline Location &Location::operator=( Location const &l ) {
   _nodeId = l._nodeId;
   _memorySpaceId = l._memorySpaceId;
   _socketId = l._socketId;
   _coreId = l._coreId;
   return *this;
}
*/

inline unsigned int Location::getClusterNode() const {
   return _clusterNode;
}
//inline void Location::setClusterNode( unsigned int clusterNode ) {
//   _clusterNode = clusterNode;
//}
inline unsigned int Location::getNumaNode() const {
   return _numaNode;
}
//inline void Location::setNumaNode( unsigned int numaNode ) {
//   _numaNode = numaNode;
//}
inline bool Location::isInNumaNode() const {
   return _inNumaNode;
}
inline unsigned int Location::getSocket() const {
   return _socket;
}
//inline void Location::setSocket( unsigned int socket ) {
//   _socket = socket;
//}
inline bool Location::isInSocket() const {
   return _inSocket;
}

/*
inline LocationDirectory::LocationDirectory() : _locations() { }
inline void LocationDirectory::initialize( unsigned int numLocs ) {
   _locations.reserve( numLocs );
}
inline Location &LocationDirectory::operator[]( unsigned int locationId ) {
   return _locations[ locationId ];
}
*/

#endif /* LOCATION_H */
