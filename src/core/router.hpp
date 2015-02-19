#ifndef NANOS_ROUTER_HPP
#define NANOS_ROUTER_HPP

#include "router_decl.hpp"

using namespace nanos;

inline Router::Router() : _lastSource( (memory_space_id_t)-1 ), _memSpaces() {
}

inline Router::~Router() {
}

inline void Router::initialize() {
   unsigned int elems = sys.getSeparateMemoryAddressSpacesCount();
   _memSpaces.resize( elems );
   for ( unsigned int idx = 0; idx < elems; idx += 1) {
      _memSpaces[ idx ] = 0;
   }
}

inline memory_space_id_t Router::getSource( memory_space_id_t destination,
      std::set<memory_space_id_t> const &locs ) {
   memory_space_id_t selected;
   //if ( locs.size() == 2 && locs.count(0) == 1 ) {
   if ( locs.size() > 1 ) {
      selected = *locs.begin();
      for (std::set<memory_space_id_t>::const_iterator it = locs.begin();
            it != locs.end(); it++ ) {
         selected = ( _memSpaces[*it] < _memSpaces[selected] ) ? *it : selected;
      }
   } else {
      selected = *locs.begin();
   }
   if ( destination != 0 && sys.getSeparateMemory( destination ).getNodeNumber() != 0 ) {
      //compute this only for cluster nodes (getNodeNumber != 0)
      _lastSource = selected;
      _memSpaces[selected] += 1;

      //if ( sys.getNetwork()->getNodeNum() == 0 ) {
      //   unsigned int min_non_zero = 0;
      //   unsigned int max_node = 0;
      //   std::cerr << "Dest: " << destination << " locs: [ ";
      //   for (std::set<memory_space_id_t>::const_iterator it = locs.begin();
      //         it != locs.end(); it++ ) {
      //      std::cerr << *it << " ";
      //   }
      //   std::cerr << "] Selected source: " << selected << " occ [ "; 
      //   for ( std::vector<unsigned int>::const_iterator vit = _memSpaces.begin(); vit != _memSpaces.end(); vit++ ) {
      //      std::cerr << *vit << " ";
      //      if ( min_non_zero == 0 && *vit != 0 ) {
      //         min_non_zero = *vit; 
      //      } else {
      //         min_non_zero = *vit != 0 ? ( min_non_zero > *vit ? *vit : min_non_zero ) : min_non_zero;
      //      }
      //      max_node = ( *vit > max_node ) ? *vit : max_node;
      //   }
      //   std::cerr << "] inb: " << (max_node - min_non_zero) << std::endl;
      //}
   }
   return selected;
}

#endif /* NANOS_ROUTER_HPP */
