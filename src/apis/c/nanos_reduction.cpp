/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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

/*! \file nanos_reduction.cpp
 *  \brief 
 */
#include "nanos_reduction.h"

NANOS_REDUCTION_INT_TYPES_DEF(add,NANOS_REDUCTION_OP_ADD)
NANOS_REDUCTION_REAL_TYPES_DEF(add,NANOS_REDUCTION_OP_ADD)
NANOS_REDUCTION_COMPLEX_TYPES_DEF(add,NANOS_REDUCTION_OP_ADD)

NANOS_REDUCTION_INT_TYPES_DEF(sub,NANOS_REDUCTION_OP_SUB)
NANOS_REDUCTION_REAL_TYPES_DEF(sub,NANOS_REDUCTION_OP_SUB)
NANOS_REDUCTION_COMPLEX_TYPES_DEF(sub,NANOS_REDUCTION_OP_SUB)

NANOS_REDUCTION_INT_TYPES_DEF(prod,NANOS_REDUCTION_OP_PROD)
NANOS_REDUCTION_REAL_TYPES_DEF(prod,NANOS_REDUCTION_OP_PROD)
NANOS_REDUCTION_COMPLEX_TYPES_DEF(prod,NANOS_REDUCTION_OP_PROD)

NANOS_REDUCTION_INT_TYPES_DEF(and,NANOS_REDUCTION_OP_AND)

NANOS_REDUCTION_INT_TYPES_DEF(or,NANOS_REDUCTION_OP_OR)

NANOS_REDUCTION_INT_TYPES_DEF(xor,NANOS_REDUCTION_OP_XOR)

NANOS_REDUCTION_INT_TYPES_DEF(land,NANOS_REDUCTION_OP_LAND)
NANOS_REDUCTION_REAL_TYPES_DEF(land,NANOS_REDUCTION_OP_LAND)
NANOS_REDUCTION_COMPLEX_TYPES_DEF(land,NANOS_REDUCTION_OP_LAND)

NANOS_REDUCTION_INT_TYPES_DEF(lor,NANOS_REDUCTION_OP_LOR)
NANOS_REDUCTION_REAL_TYPES_DEF(lor,NANOS_REDUCTION_OP_LOR)
NANOS_REDUCTION_COMPLEX_TYPES_DEF(lor,NANOS_REDUCTION_OP_LOR)

NANOS_REDUCTION_INT_TYPES_DEF(max,NANOS_REDUCTION_OP_MAX)
NANOS_REDUCTION_REAL_TYPES_DEF(max,NANOS_REDUCTION_OP_MAX)

NANOS_REDUCTION_INT_TYPES_DEF(min,NANOS_REDUCTION_OP_MIN)
NANOS_REDUCTION_REAL_TYPES_DEF(min,NANOS_REDUCTION_OP_MIN)

NANOS_REDUCTION_CLEANUP_DEF(char, char)
NANOS_REDUCTION_CLEANUP_DEF(uchar, unsigned char)
NANOS_REDUCTION_CLEANUP_DEF(schar, signed char)
NANOS_REDUCTION_CLEANUP_DEF(short, short)
NANOS_REDUCTION_CLEANUP_DEF(ushort, unsigned short)
NANOS_REDUCTION_CLEANUP_DEF(int, int)
NANOS_REDUCTION_CLEANUP_DEF(uint, unsigned int)
NANOS_REDUCTION_CLEANUP_DEF(long, long)
NANOS_REDUCTION_CLEANUP_DEF(ulong, unsigned long)
NANOS_REDUCTION_CLEANUP_DEF(longlong, long long)
NANOS_REDUCTION_CLEANUP_DEF(ulonglong, unsigned long long )
NANOS_REDUCTION_CLEANUP_DEF(_Bool, _Bool)
NANOS_REDUCTION_CLEANUP_DEF(float, float)
NANOS_REDUCTION_CLEANUP_DEF(double, double)
NANOS_REDUCTION_CLEANUP_DEF(longdouble, long double)
NANOS_REDUCTION_CLEANUP_DEF(cfloat, _Complex float)
NANOS_REDUCTION_CLEANUP_DEF(cdouble, _Complex double)
NANOS_REDUCTION_CLEANUP_DEF(clongdouble, _Complex long double)

