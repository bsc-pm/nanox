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

#ifndef NANOS_ROUTER_HPP
#define NANOS_ROUTER_HPP

#include "router_decl.hpp"

namespace nanos {

inline Router::Router() : _lastSource( (memory_space_id_t)-1 ), _memSpaces() {
}

inline Router::~Router() {
}

inline void Router::initialize() {
   unsigned int elems = sys.getSeparateMemoryAddressSpacesCount() + 1; //remote nodes + master node
   _memSpaces.resize( elems );
   for ( unsigned int idx = 0; idx < elems; idx += 1) {
      _memSpaces[ idx ] = 0;
   }
}

inline memory_space_id_t Router::getSource( memory_space_id_t destination,
      std::set<memory_space_id_t> const &locs ) {
   memory_space_id_t selected;
   unsigned int destination_node = destination != 0 ? sys.getSeparateMemory( destination ).getNodeNumber() : 0;
   if ( locs.size() > 1 ) {

      //clasify the locations in remote (remote nodes) or local (host memory or accelerators)
      memory_space_id_t tmp_locations[ locs.size() ];
      int local_locations_idx = 0;
      int remote_locations_idx = locs.size()-1;
      for (std::set<memory_space_id_t>::const_iterator it = locs.begin();
            it != locs.end(); it++ ) {
         if ( *it == 0 || sys.getSeparateMemory( *it ).getNodeNumber() ) {
            tmp_locations[local_locations_idx] = *it;
            local_locations_idx += 1;
         } else {
            tmp_locations[remote_locations_idx] = *it;
            remote_locations_idx -= 1;
         }
      }

      if ( destination == 0 || destination_node == 0 ) {
         //if destination is a local address space, priorize getting the data from a local location
         if ( local_locations_idx == 0 ) {
            //no local locations available
            selected = tmp_locations[remote_locations_idx + 1];
            for ( int idx = remote_locations_idx + 1; idx < (int) locs.size(); idx += 1 ) {
               selected = ( _memSpaces[ tmp_locations[idx] ] < _memSpaces[selected] ) ? tmp_locations[idx] : selected;
            }
         } else if ( local_locations_idx == 1 ) {
            //local location available and there is only one, use it
            selected = tmp_locations[local_locations_idx - 1];
         } else {
            //more than one location available
            selected = tmp_locations[0];
            for ( int idx = 0; idx < local_locations_idx; idx += 1 ) {
               selected = ( _memSpaces[ tmp_locations[idx] ] < _memSpaces[selected] ) ? tmp_locations[idx] : selected;
            }
         }
      } else {
         // remote destination, use 0 or remote nodes, not local accelerators if possible
         selected = tmp_locations[0];
         for ( int idx = remote_locations_idx + 1; idx < (int)locs.size(); idx += 1 ) {
            selected = ( _memSpaces[ tmp_locations[idx] ] < _memSpaces[selected] ) ? tmp_locations[idx] : selected;
         }
      }
   } else {
      selected = *locs.begin();
   }

   unsigned int selected_node = selected != 0 ? sys.getSeparateMemory( selected ).getNodeNumber() : 0;
   // To keep cluster balance we want to count transferences from node 0 to remote and NOT from node 0 to local accel
   if ( !( selected_node == 0 && destination_node == 0 ) ) {
      _lastSource = selected;
      _memSpaces[selected] += 1;
   }
   return selected;
}

} // namespace nanos

#endif /* NANOS_ROUTER_HPP */
