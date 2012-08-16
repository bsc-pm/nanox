#ifndef LOCATION_H
#define LOCATION_H

#include "location_decl.hpp"

using namespace nanos;

inline Location::Location() : _nodeId( (unsigned int) -1 ), _memorySpaceId( ( unsigned int ) -1 ), _socketId( ( unsigned int ) -1 ), _coreId( (unsigned int) -1 ) { }
inline Location::Location( Location const &l) : _nodeId( l._nodeId ), _memorySpaceId( l._memorySpaceId ), _socketId( l._socketId ), _coreId( l._coreId ) { }
inline Location &Location::operator=( Location const &l ) {
   _nodeId = l._nodeId;
   _memorySpaceId = l._memorySpaceId;
   _socketId = l._socketId;
   _coreId = l._coreId;
   return *this;
}

inline unsigned int Location::getNodeId() const {
   return _nodeId;
}
inline void Location::setNodeId( unsigned int nodeId ) {
   _nodeId = nodeId;
}
inline unsigned int Location::getMemorySpaceId() const {
   return _memorySpaceId;
}
inline void Location::setMemorySpaceId( unsigned int memId ) {
   _memorySpaceId = memId;
}
inline unsigned int Location::getSocketId() const {
   return _socketId;
}
inline void Location::setSocketId( unsigned int socketId ) {
   _socketId = socketId;
}
inline unsigned int Location::getCoreId() const {
   return _coreId;
}
inline void Location::setCoreId( unsigned int coreId ) {
   _coreId = coreId;
}

inline LocationDirectory::LocationDirectory() : _locations() { }
inline void LocationDirectory::initialize( unsigned int numLocs ) {
   _locations.reserve( numLocs );
}
inline Location &LocationDirectory::operator[]( unsigned int locationId ) {
   return _locations[ locationId ];
}

#endif /* LOCATION_H */
