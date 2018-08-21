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

#ifndef VERSION_HPP
#define VERSION_HPP
#include "system_decl.hpp"

namespace nanos {

inline Version::Version() : _version( 0 ) {
}

inline Version::Version( Version const & ver ) : _version( ver._version ) {
}

inline Version::Version( unsigned int v ) : _version( v ) {
}

inline Version::~Version() {
}

inline Version &Version::operator=( Version const & ver ) {
   _version = ver._version;
   return *this;
}

inline unsigned int Version::getVersion() const {
   return _version;
}

inline unsigned int Version::getVersion( bool increaseVersion ) {
   unsigned int current_version = _version;
   if ( increaseVersion ) {
      _version += 1;
   }
   return current_version;
}

inline void Version::setVersion( unsigned int version ) {
   if ( version < _version ) {
      (*myThread->_file) << "WARNING not version increase " << _version << " => " << version << std::endl;
      printBt( *(myThread->_file) );
   }
   _version = version;
}

inline void Version::resetVersion() {
   _version = 0;
}

} // namespace nanos

#endif /* VERSION_HPP */
