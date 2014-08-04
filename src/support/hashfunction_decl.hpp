#ifndef HASHFUNCTION_DECL
#define HASHFUNCTION_DECL

#include <stdint.h>

namespace nanos {

#define mix(a,b,c) \
{ \
  a -= b; a -= c; a ^= ( c >> 13 ); \
  b -= c; b -= a; b ^= ( a << 8 ); \
  c -= a; c -= b; c ^= ( b >> 13 ); \
  a -= b; a -= c; a ^= ( c >> 12 ); \
  b -= c; b -= a; b ^= ( a << 16 ); \
  c -= a; c -= b; c ^= ( b >> 5 ); \
  a -= b; a -= c; a ^= ( c >> 3 ); \
  b -= c; b -= a; b ^= ( a << 10 ); \
  c -= a; c -= b; c ^= ( b >> 15 ); \
}

unsigned int jen_hash ( uint64_t value );

unsigned int jen_hash ( uint64_t value )
{
   unsigned int len = sizeof(void *);
   unsigned int a, b, c;

   a = 0x9e3779b9 + ((unsigned int) (value & 0xffffffff));
   b = 0x9e3779b9 + ((unsigned int) (value >> 32));
   c = len; /* initval */

   mix ( a, b, c );

   return c;
}

}
#endif
