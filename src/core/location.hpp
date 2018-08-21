/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
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
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/

#ifndef LOCATION_H
#define LOCATION_H

#include "location_decl.hpp"

namespace nanos {

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

} // namespace nanos

#endif /* LOCATION_H */
