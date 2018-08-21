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
