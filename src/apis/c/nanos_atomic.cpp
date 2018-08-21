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

/*! \file nanos_atomic.cpp
 *  \brief 
 */
#include "nanos_atomic.h"

#include "atomic.hpp"
#include "lock.hpp"

#include <stdint.h>
#include <math.h>
#include <signal.h>
#include <complex.h>

namespace {
nanos::Lock update_lock;
}

extern "C"
{
#ifndef HAVE_SYNC_BOOL_COMPARE_AND_SWAP_8
    bool __sync_bool_compare_and_swap_8 (volatile void* v_ptr, long long unsigned oldval, long long unsigned newval)
    {
        LockBlock l(update_lock);
		volatile long long unsigned* ptr = reinterpret_cast<volatile long long unsigned*>(v_ptr);
        if (*ptr == oldval)
        {
            *ptr = newval;
            return true;
        }
        return false;
    }
#endif

#ifndef HAVE_SYNC_ADD_AND_FETCH_8
    unsigned long long __sync_add_and_fetch_8 (volatile void *v_ptr, unsigned long long value)
    {
        LockBlock l(update_lock);
		volatile long long unsigned* ptr = reinterpret_cast<volatile long long unsigned*>(v_ptr);
        *ptr += value;

		return *ptr;
    }
#endif

#ifndef HAVE_SYNC_SUB_AND_FETCH_8
    unsigned long long __sync_sub_and_fetch_8 (volatile void *v_ptr, unsigned long long value)
    {
        LockBlock l(update_lock);
		volatile long long unsigned* ptr = reinterpret_cast<volatile long long unsigned*>(v_ptr);
        *ptr -= value;

		return *ptr;
    }
#endif

#ifndef HAVE_SYNC_AND_AND_FETCH_8
    unsigned long long __sync_and_and_fetch_8 (volatile void *v_ptr, unsigned long long value)
    {
        LockBlock l(update_lock);
		volatile long long unsigned* ptr = reinterpret_cast<volatile long long unsigned*>(v_ptr);
        *ptr &= value;

		return *ptr;
    }
#endif

#ifndef HAVE_SYNC_OR_AND_FETCH_8
    unsigned long long __sync_or_and_fetch_8 (volatile void *v_ptr, unsigned long long value)
    {
        LockBlock l(update_lock);
		volatile long long unsigned* ptr = reinterpret_cast<volatile long long unsigned*>(v_ptr);
        *ptr |= value;

		return *ptr;
    }
#endif

#ifndef HAVE_SYNC_XOR_AND_FETCH_8
    unsigned long long __sync_xor_and_fetch_8 (volatile void *v_ptr, unsigned long long value)
    {
        LockBlock l(update_lock);
		volatile long long unsigned* ptr = reinterpret_cast<volatile long long unsigned*>(v_ptr);
        *ptr ^= value;

		return *ptr;
    }
#endif
}

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

#ifdef __GXX_EXPERIMENTAL_CXX0X__
extern "C"
{
    // For some reason g++, #include <complex.h> in C++2011 does not include C99 complex.h when in C++98 does
    // We repeat the declarations here so they link with a C99 -lm
#pragma GCC diagnostic ignored "-Wredundant-decls"
    extern _Complex float cpowf(_Complex float, _Complex float) throw();
    extern _Complex double cpow(_Complex double, _Complex double) throw ();
    extern _Complex long double cpowl(_Complex long double, _Complex long double) throw ();
#pragma GCC diagnostic error "-Wredundant-decls"
}
#endif

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

// Implementation of each atomic op for integer types
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
#define NANOS_PERFORM_ATOMIC_OP_INT_max(type, x, y)     NANOS_CAS_ATOMIC(type, x, y, max)
#define NANOS_PERFORM_ATOMIC_OP_INT_min(type, x, y)     NANOS_CAS_ATOMIC(type, x, y, min)
#define NANOS_PERFORM_ATOMIC_OP_INT_eq(type, x, y)     NANOS_CAS_ATOMIC(type, x, y, eq)
#define NANOS_PERFORM_ATOMIC_OP_INT_neq(type, x, y)     NANOS_CAS_ATOMIC(type, x, y, neq)

#define NANOS_PERFORM_ATOMIC_OP_INT_assig(type, x, y)   NANOS_CAS_ATOMIC(type, x, y, assig)

// Implementation of each atomic op for floating types
#define NANOS_PERFORM_ATOMIC_OP_FLOAT(type_name, type, op, x, y)   NANOS_PERFORM_ATOMIC_OP_FLOAT_##type_name(type, op, x, y)

#define NANOS_PERFORM_ATOMIC_OP_FLOAT_float(type, op, x, y)      NANOS_CAS_ATOMIC(type, x, y, op)
#define NANOS_PERFORM_ATOMIC_OP_FLOAT_double(type, op, x, y)     NANOS_CAS_ATOMIC(type, x, y, op)
#ifdef HAVE_INT128_T
 #define NANOS_PERFORM_ATOMIC_OP_FLOAT_ldouble(type, op, x, y)   NANOS_CAS_ATOMIC(type, x, y, op)
#else
 #define NANOS_PERFORM_ATOMIC_OP_FLOAT_ldouble(type, op, x, y)   NANOS_LOCK_UPDATE(type, x, y, op)
#endif

// Implementation of each atomic op for complex floating types
#define NANOS_PERFORM_ATOMIC_OP_COMPLEX(type_name, type, op, x, y)   NANOS_PERFORM_ATOMIC_OP_COMPLEX_##type_name(type, op, x, y)

#define NANOS_PERFORM_ATOMIC_OP_COMPLEX_cfloat(type, op, x, y)   NANOS_LOCK_UPDATE(type, x, y, op)
#define NANOS_PERFORM_ATOMIC_OP_COMPLEX_cdouble(type, op, x, y)  NANOS_LOCK_UPDATE(type, x, y, op)
#define NANOS_PERFORM_ATOMIC_OP_COMPLEX_cldouble(type, op, x, y) NANOS_LOCK_UPDATE(type, x, y, op)

// Expression used by each operation in the CAS (compare-and-swap) template below
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
#define NANOS_CAS_BIN_OP_max(x, y) (x > y ? x : y)
#define NANOS_CAS_BIN_OP_min(x, y) (x < y ? x : y)
#define NANOS_CAS_BIN_OP_eq(x, y) (x == y)
#define NANOS_CAS_BIN_OP_neq(x, y) (x != y)

#define NANOS_CAS_BIN_OP_assig(x, y) y

// Template for CAS (compare-and-swap)
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

// Template for lock implementation
#define NANOS_LOCK_UPDATE(type, x, y, op) \
{ \
    nanos::LockBlock b(update_lock); \
    (*x) = NANOS_CAS_BIN_OP_##op((*x), y); \
}

// Integral types
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
    NANOS_ATOMIC_DEF_INT_OP(op, ulonglong, unsigned long long) \
    \
    NANOS_ATOMIC_DEF_INT_OP(op, bytebool, signed char) \
    NANOS_ATOMIC_DEF_INT_OP(op, shortbool, short) \
    NANOS_ATOMIC_DEF_INT_OP(op, intbool, int) \
    NANOS_ATOMIC_DEF_INT_OP(op, longbool, long int) \
    NANOS_ATOMIC_DEF_INT_OP(op, longlongbool, long long int)

// Float types
#define NANOS_ATOMIC_FLOAT_OP(op) \
    NANOS_ATOMIC_DEF_FLOAT_OP(op, float, float) \
    NANOS_ATOMIC_DEF_FLOAT_OP(op, double, double) \
    NANOS_ATOMIC_DEF_FLOAT_OP(op, ldouble, long double)

// Complex float types
#define NANOS_ATOMIC_COMPLEX_OP(op) \
    NANOS_ATOMIC_DEF_COMPLEX_OP(op, cfloat, _Complex float) \
    NANOS_ATOMIC_DEF_COMPLEX_OP(op, cdouble, _Complex double) \
    NANOS_ATOMIC_DEF_COMPLEX_OP(op, cldouble, _Complex long double)

// All types
#define NANOS_ATOMIC_ALL_OP(op) \
    NANOS_ATOMIC_INT_OP(op) \
    NANOS_ATOMIC_FLOAT_OP(op) \
    NANOS_ATOMIC_COMPLEX_OP(op)

#ifndef __MIC__
// Emit implementation of atomic (op X types), for each op and type
ATOMIC_OPS
#endif
