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

#ifndef _NANOS_ADDRESS_H
#define _NANOS_ADDRESS_H

#include "address_decl.hpp"

namespace nanos {

inline const Address & Address::operator= ( const Address &obj )
{
   _address = obj._address;
   return *this;
}

inline const Address::TargetType& Address::operator() () const
{
   return _address;
}

inline bool Address::operator== ( const Address &obj ) const
{
   return _address == obj._address;
}

inline bool Address::overlap ( const BaseDependency &obj ) const
{
   const Address& address( static_cast<const Address&>( obj ) );
   return _address == address._address;
}

inline bool Address::operator< ( const Address &obj ) const
{
   return _address < obj._address;
}

inline BaseDependency* Address::clone() const
{
   return new Address( _address );
}

inline void * Address::getAddress () const
{
   return _address;
}

} // namespace nanos

#endif
