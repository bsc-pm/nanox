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


#ifndef _NANOS_BIT_COUNTER
#define _NANOS_BIT_COUNTER

#include "bitcounter_decl.hpp"

namespace nanos {

template<typename T, int BITS>
bool inline BitCounter<T,BITS>::hasMoreThanOneOne(T value)
{
   T lowMask = 0;
   lowMask--;
   lowMask = lowMask >> (BITS >> 1);
   T highPart = value >> (BITS >> 1);
   T lowPart = value & lowMask;
   return (lowPart & highPart) | BitCounter<T, (BITS >> 1)>::hasMoreThanOneOne(lowPart ^ highPart);
}

template<typename T>
bool inline BitCounter<T, 2>::hasMoreThanOneOne(T value)
{
   T highPart = value >> 1;
   T lowPart = value & 1;
   return lowPart & highPart;
}

} // namespace nanos

#endif // _NANOS_BIT_COUNTER
