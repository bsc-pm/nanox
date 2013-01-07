#ifndef VERSION_HPP
#define VERSION_HPP
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

inline void Version::setVersion( unsigned int version ) {
   _version = version;
}
#endif /* VERSION_HPP */
