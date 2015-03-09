#ifndef VERSION_HPP
#define VERSION_HPP
#include "system_decl.hpp"
using namespace nanos;

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
#endif /* VERSION_HPP */
