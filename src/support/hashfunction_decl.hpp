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

} // namespace nanos

#endif
