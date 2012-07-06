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

#include "nanos_c_api_macros.h"
#include "nanos_atomic.h"

#include "atomic.hpp"

#include <stdint.h>
#include <math.h>
#include <signal.h>
#include <complex.h>

template <int N>
struct suitable_integer_type { };

#ifdef HAVE_INT128_T
template <>
struct suitable_integer_type<16> { typedef __int128 T; };
#endif

template <>
struct suitable_integer_type<8> { typedef uint64_t T; };

template <>
struct suitable_integer_type<4> { typedef uint32_t T; };

template <>
struct suitable_integer_type<2> { typedef uint16_t T; };

template <>
struct suitable_integer_type<1> { typedef uint8_t T; };

template <typename T_>
struct atomic_type_trait
{
    typedef typename suitable_integer_type<sizeof(T_)>::T T;
};


namespace
{
    template <typename T>
        inline T generic_pow(T a, T b)
        {
            // Not implemented yet
            raise(SIGABRT);
            return T(0);
        }

    template <>
        inline float generic_pow(float a, float b) { return ::powf(a, b); }

    template <>
        inline double generic_pow(double a, double b) { return ::pow(a, b); }

    template <>
        inline long double generic_pow(long double a, long double b) { return ::powl(a, b); }

    template <>
        inline _Complex float generic_pow(_Complex float a, _Complex float b) { return ::cpowf(a, b); }

    template <>
        inline _Complex double generic_pow(_Complex double a, _Complex double b) { return ::cpow(a, b); }

    template <>
        inline _Complex long double generic_pow(_Complex long double a, _Complex long double b) { return ::cpowl(a, b); }
}


// void f(volatile type* x, type y)
// {
// x = x op y
// }

#define NANOS_ATOMIC_DEF_INT_OP(op, type_name, type) \
NANOS_API_DEF(void, nanos_atomic_##op##_##type_name, (volatile type * x, type y)) \
{ \
    NANOS_PERFORM_ATOMIC_OP_INT_##op(type, x, y); \
}

#define NANOS_ATOMIC_DEF_FLOAT_OP(op, type_name, type) \
NANOS_API_DEF(void, nanos_atomic_##op##_##type_name, (volatile type * x, type y)) \
{ \
    NANOS_PERFORM_ATOMIC_OP_FLOAT(type_name, type, op, x, y); \
}

#define NANOS_ATOMIC_DEF_COMPLEX_OP(op, type_name, type) \
NANOS_API_DEF(void, nanos_atomic_##op##_##type_name, (volatile type * x, type y)) \
{ \
    NANOS_PERFORM_ATOMIC_OP_COMPLEX(type_name, type, op, x, y); \
}

#define NANOS_PERFORM_ATOMIC_OP_INT_add(_, x, y)        __sync_add_and_fetch(x, y)
#define NANOS_PERFORM_ATOMIC_OP_INT_sub(_, x, y)        __sync_sub_and_fetch(x, y)
#define NANOS_PERFORM_ATOMIC_OP_INT_mul(type, x, y)     NANOS_CAS_ATOMIC(type, x, y, mul)
#define NANOS_PERFORM_ATOMIC_OP_INT_div(type, x, y)     NANOS_CAS_ATOMIC(type, x, y, div)
#define NANOS_PERFORM_ATOMIC_OP_INT_pow(type, x, y)     NANOS_CAS_ATOMIC(type, x, y, pow)
#define NANOS_PERFORM_ATOMIC_OP_INT_mod(type, x, y)     NANOS_CAS_ATOMIC(type, x, y, mod)
#define NANOS_PERFORM_ATOMIC_OP_INT_shl(type, x, y)     NANOS_CAS_ATOMIC(type, x, y, shl)
#define NANOS_PERFORM_ATOMIC_OP_INT_shr(type, x, y)     NANOS_CAS_ATOMIC(type, x, y, shr)
#define NANOS_PERFORM_ATOMIC_OP_INT_land(type, x, y)    NANOS_CAS_ATOMIC(type, x, y, land)
#define NANOS_PERFORM_ATOMIC_OP_INT_lor(type, x, y)     NANOS_CAS_ATOMIC(type, x, y, lor)
#define NANOS_PERFORM_ATOMIC_OP_INT_band(type, x, y)    __sync_and_and_fetch(x, y)
#define NANOS_PERFORM_ATOMIC_OP_INT_bor(type, x, y)     __sync_or_and_fetch(x, y)
#define NANOS_PERFORM_ATOMIC_OP_INT_bxor(type, x, y)    __sync_xor_and_fetch(x, y)

#define NANOS_PERFORM_ATOMIC_OP_INT_assig(type, x, y)   NANOS_CAS_ATOMIC(type, x, y, assig)

#define NANOS_PERFORM_ATOMIC_OP_FLOAT(type_name, type, op, x, y)   NANOS_PERFORM_ATOMIC_OP_FLOAT_##type_name(type, op, x, y)
#define NANOS_PERFORM_ATOMIC_OP_FLOAT(type_name, type, op, x, y)   NANOS_PERFORM_ATOMIC_OP_FLOAT_##type_name(type, op, x, y)
#define NANOS_PERFORM_ATOMIC_OP_FLOAT(type_name, type, op, x, y)   NANOS_PERFORM_ATOMIC_OP_FLOAT_##type_name(type, op, x, y)
#define NANOS_PERFORM_ATOMIC_OP_FLOAT(type_name, type, op, x, y)   NANOS_PERFORM_ATOMIC_OP_FLOAT_##type_name(type, op, x, y)
#define NANOS_PERFORM_ATOMIC_OP_FLOAT(type_name, type, op, x, y)   NANOS_PERFORM_ATOMIC_OP_FLOAT_##type_name(type, op, x, y)

#define NANOS_PERFORM_ATOMIC_OP_FLOAT_float(type, op, x, y)      NANOS_CAS_ATOMIC(type, x, y, op)
#define NANOS_PERFORM_ATOMIC_OP_FLOAT_double(type, op, x, y)     NANOS_CAS_ATOMIC(type, x, y, op)
#ifdef HAVE_INT128_T
 #define NANOS_PERFORM_ATOMIC_OP_FLOAT_ldouble(type, op, x, y)   NANOS_CAS_ATOMIC(type, x, y, op)
#else
 #define NANOS_PERFORM_ATOMIC_OP_FLOAT_ldouble(type, op, x, y)   NANOS_LOCK_UPDATE(type, x, y, op)
#endif

#define NANOS_PERFORM_ATOMIC_OP_COMPLEX(type_name, type, op, x, y)   NANOS_PERFORM_ATOMIC_OP_COMPLEX_##type_name(type, op, x, y)
#define NANOS_PERFORM_ATOMIC_OP_COMPLEX(type_name, type, op, x, y)   NANOS_PERFORM_ATOMIC_OP_COMPLEX_##type_name(type, op, x, y)
#define NANOS_PERFORM_ATOMIC_OP_COMPLEX(type_name, type, op, x, y)   NANOS_PERFORM_ATOMIC_OP_COMPLEX_##type_name(type, op, x, y)
#define NANOS_PERFORM_ATOMIC_OP_COMPLEX(type_name, type, op, x, y)   NANOS_PERFORM_ATOMIC_OP_COMPLEX_##type_name(type, op, x, y)
#define NANOS_PERFORM_ATOMIC_OP_COMPLEX(type_name, type, op, x, y)   NANOS_PERFORM_ATOMIC_OP_COMPLEX_##type_name(type, op, x, y)

#define NANOS_PERFORM_ATOMIC_OP_COMPLEX_cfloat(type, op, x, y)   NANOS_LOCK_UPDATE(type, x, y, op)
#define NANOS_PERFORM_ATOMIC_OP_COMPLEX_cdouble(type, op, x, y)  NANOS_LOCK_UPDATE(type, x, y, op)
#define NANOS_PERFORM_ATOMIC_OP_COMPLEX_cldouble(type, op, x, y) NANOS_LOCK_UPDATE(type, x, y, op)

#define NANOS_CAS_BIN_OP_add(x, y) (x + y)
#define NANOS_CAS_BIN_OP_sub(x, y) (x - y)
#define NANOS_CAS_BIN_OP_mul(x, y) (x * y)
#define NANOS_CAS_BIN_OP_div(x, y) (x / y)
#define NANOS_CAS_BIN_OP_mod(x, y) (x % y)
#define NANOS_CAS_BIN_OP_shl(x, y) (x << y)
#define NANOS_CAS_BIN_OP_shr(x, y) (x >> y)
#define NANOS_CAS_BIN_OP_land(x, y) (x && y)
#define NANOS_CAS_BIN_OP_lor(x, y) (x || y)
#define NANOS_CAS_BIN_OP_pow(x, y) generic_pow(x, y)

#define NANOS_CAS_BIN_OP_assig(x, y) y

#define NANOS_CAS_ATOMIC(type, x, y, op) \
{ \
    typedef atomic_type_trait<type>::T atomic_int_t; \
    union U { type val; atomic_int_t v; } old, new_; \
    do { \
        old.val = (*x); \
        new_.val = NANOS_CAS_BIN_OP_##op(old.val, y); \
        __sync_synchronize(); \
    } while (!__sync_bool_compare_and_swap((atomic_int_t*)x, old.v, new_.v)); \
}

namespace {
nanos::Lock update_lock;
}

#define NANOS_LOCK_UPDATE(type, x, y, op) \
{ \
    nanos::LockBlock b(update_lock); \
    (*x) = NANOS_CAS_BIN_OP_##op((*x), y); \
}

#define NANOS_ATOMIC_INT_OP(op) \
    NANOS_ATOMIC_DEF_INT_OP(op, schar, signed char) \
    NANOS_ATOMIC_DEF_INT_OP(op, short, short) \
    NANOS_ATOMIC_DEF_INT_OP(op, int, int) \
    NANOS_ATOMIC_DEF_INT_OP(op, long, long) \
    NANOS_ATOMIC_DEF_INT_OP(op, longlong, long long) \
    NANOS_ATOMIC_DEF_INT_OP(op, uchar, unsigned char) \
    NANOS_ATOMIC_DEF_INT_OP(op, ushort, unsigned short int) \
    NANOS_ATOMIC_DEF_INT_OP(op, uint, unsigned int) \
    NANOS_ATOMIC_DEF_INT_OP(op, ulong, unsigned long) \

#define NANOS_ATOMIC_FLOAT_OP(op) \
    NANOS_ATOMIC_DEF_FLOAT_OP(op, float, float) \
    NANOS_ATOMIC_DEF_FLOAT_OP(op, double, double) \
    NANOS_ATOMIC_DEF_FLOAT_OP(op, ldouble, long double)

#define NANOS_ATOMIC_COMPLEX_OP(op) \
    NANOS_ATOMIC_DEF_COMPLEX_OP(op, cfloat, _Complex float) \
    NANOS_ATOMIC_DEF_COMPLEX_OP(op, cdouble, _Complex double) \
    NANOS_ATOMIC_DEF_COMPLEX_OP(op, cldouble, _Complex long double)

#define NANOS_ATOMIC_ALL_OP(op) \
    NANOS_ATOMIC_INT_OP(op) \
    NANOS_ATOMIC_FLOAT_OP(op) \
    NANOS_ATOMIC_COMPLEX_OP(op)

#ifndef __MIC__
ATOMIC_OPS
#endif
