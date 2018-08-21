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

/*
	Cell/SMP Superscalar (CellSs/SMPSs): Easy programming the Cell BE/Shared Memory Processors
	Copyright (C) 2008 Barcelona Supercomputing Center - Centro Nacional de Supercomputacion
	
	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.
	
	This library is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
	Lesser General Public License for more details.
	
	You should have received a copy of the GNU Lesser General Public
	License along with this library; if not, write to the Free Software
	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
	
	The GNU Lesser General Public License is contained in the file COPYING.
*/


#ifndef _NANOS_BIT_COUNTER
#define _NANOS_BIT_COUNTER


namespace nanos {


template<typename T, int BITS = sizeof(T)*8>
class BitCounter {
   public:
      static bool __attribute__((always_inline)) hasMoreThanOneOne(T value)
         {
            T lowMask = 0;
            lowMask--;
            lowMask = lowMask >> (BITS >> 1);
            T highPart = value >> (BITS >> 1);
            T lowPart = value & lowMask;
            return (lowPart & highPart) | BitCounter<T, (BITS >> 1)>::hasMoreThanOneOne(lowPart ^ highPart);
         }

};


template<typename T>
class BitCounter<T, 2> {
   public:
      static bool __attribute__((always_inline)) hasMoreThanOneOne(T value)
         {
            T highPart = value >> 1;
            T lowPart = value & 1;
            return lowPart & highPart;
         }

};


} // namespace nanos

#endif // _NANOS_BIT_COUNTER
