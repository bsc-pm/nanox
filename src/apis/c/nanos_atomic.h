/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
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

#ifndef _NANOS_ATOMIC_H
#define _NANOS_ATOMIC_H

#include "nanos_c_api_macros.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define ATOMIC_OPS \
    NANOS_ATOMIC_ALL_OP(assig) \
    NANOS_ATOMIC_ALL_OP(add) \
    NANOS_ATOMIC_ALL_OP(sub) \
    NANOS_ATOMIC_ALL_OP(mul) \
    NANOS_ATOMIC_ALL_OP(div) \
    NANOS_ATOMIC_ALL_OP(pow) \
    NANOS_ATOMIC_INT_OP(mod) \
    NANOS_ATOMIC_INT_OP(shl) \
    NANOS_ATOMIC_INT_OP(shr) \
    NANOS_ATOMIC_INT_OP(land) \
    NANOS_ATOMIC_INT_OP(lor) \
    NANOS_ATOMIC_INT_OP(band) \
    NANOS_ATOMIC_INT_OP(bor) \
    NANOS_ATOMIC_INT_OP(bxor)

#define NANOS_ATOMIC_DECL_OP(op, type_name, type) \
    NANOS_API_DECL(void, nanos_atomic_##op##_##type_name, (volatile type *, type));

#define NANOS_ATOMIC_INT_OP(op) \
    NANOS_ATOMIC_DECL_OP(op, schar, signed char) \
    NANOS_ATOMIC_DECL_OP(op, short, short) \
    NANOS_ATOMIC_DECL_OP(op, int, int) \
    NANOS_ATOMIC_DECL_OP(op, long, long) \
    NANOS_ATOMIC_DECL_OP(op, longlong, long long) \
    NANOS_ATOMIC_DECL_OP(op, uchar, unsigned char) \
    NANOS_ATOMIC_DECL_OP(op, ushort, unsigned short int) \
    NANOS_ATOMIC_DECL_OP(op, uint, unsigned int) \
    NANOS_ATOMIC_DECL_OP(op, ulong, unsigned long) \


#define NANOS_ATOMIC_FLOAT_OP(op) \
    NANOS_ATOMIC_DECL_OP(op, float, float) \
    NANOS_ATOMIC_DECL_OP(op, double, double) \
    NANOS_ATOMIC_DECL_OP(op, ldouble, long double)

#define NANOS_ATOMIC_COMPLEX_OP(op) \
    NANOS_ATOMIC_DECL_OP(op, cfloat, _Complex float) \
    NANOS_ATOMIC_DECL_OP(op, cdouble, _Complex double) \
    NANOS_ATOMIC_DECL_OP(op, cldouble, _Complex long double)

#define NANOS_ATOMIC_ALL_OP(op) \
    NANOS_ATOMIC_INT_OP(op) \
    NANOS_ATOMIC_FLOAT_OP(op) \
    NANOS_ATOMIC_COMPLEX_OP(op)

ATOMIC_OPS

#undef NANOS_ATOMIC_ALL_OP
#undef NANOS_ATOMIC_COMPLEX_OP
#undef NANOS_ATOMIC_FLOAT_OP
#undef NANOS_ATOMIC_INT_OP

#ifdef __cplusplus
}
#endif

#endif // _NANOS_ATOMIC_H
