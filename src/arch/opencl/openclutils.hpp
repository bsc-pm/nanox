/*************************************************************************************/
/*      Copyright 2013 Barcelona Supercomputing Center                               */
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
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

#ifndef _NANOS_OpenCL_UTILS
#define _NANOS_OpenCL_UTILS

#include <unistd.h>

namespace nanos {
namespace ext {

inline size_t getPageSize() { return getpagesize(); }

inline size_t roundUpToPageSize( size_t size )
{
   size_t pageSize = getPageSize(),
          extra = size % pageSize;

   if( extra != 0)
     size += pageSize - extra;

   return size;
}

inline uint32_t log2( uint32_t x )
{
   union {
      uint32_t uint[2];
      double dbl;
   } temp;

   temp.uint[__FLOAT_WORD_ORDER == LITTLE_ENDIAN] = 0x43300000;
   temp.uint[__FLOAT_WORD_ORDER != LITTLE_ENDIAN] = x;
   temp.dbl -= 4503599627370496.0;

   return ( temp.uint[__FLOAT_WORD_ORDER == LITTLE_ENDIAN] >> 20 ) - 0x3FF;
}

inline uint32_t gnuHash( const char *str )
{
   uint32_t h = 5381;

   for ( unsigned char c = *str; c != '\0'; c = *++str )
      h = h * 33 + c;

   return h;
}

inline uint32_t gnuHash( const char *str, const char *end )
{
   uint32_t h = 5381;

   for ( unsigned char c = *str; str != end; c = *++str )
      h = h * 33 + c;

   return h;
}

} // End namespace ext.
} // End namespace nanos.

#endif // _NANOS_OpenCL_UTILS
