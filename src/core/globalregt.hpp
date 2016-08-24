#ifndef GLOBALREGT_HPP
#define GLOBALREGT_HPP

#include "globalregt_decl.hpp"

namespace nanos {

inline global_reg_t::global_reg_t( reg_t r, reg_key_t k ) : id( r ), key( k ) {
}

inline global_reg_t::global_reg_t( reg_t r, const_reg_key_t k ) : id( r ), ckey( k ) {
}

inline global_reg_t::global_reg_t() : id( 0 ), key( NULL ) {
}

inline bool global_reg_t::operator<( global_reg_t const &reg ) const {
   bool result;
   if ( key < reg.key )
      result = true;
   else if ( reg.key < key )
      result = false;
   else result = ( id < reg.id );
   return result;
}

inline bool global_reg_t::operator!=( global_reg_t const &reg ) const {
   bool result = false;
   if ( key != reg.key )
      result = true;
   else if ( reg.key != key )
      result = true;
   return result;
}

inline memory_space_id_t global_reg_t::getFirstLocation() const {
   return RegionDirectory::getFirstLocation( key, id );
}

} // namespace nanos

#endif
