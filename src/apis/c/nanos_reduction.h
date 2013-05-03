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
/*! \file nanos_reduction.h
 *  \brief 
 */
#include "nanos-int.h"

#ifndef _NANOS_REDUCTION_H_
#define _NANOS_REDUCTION_H_

#ifdef __cplusplus
extern "C"
{
#endif

#define NANOS_REDUCTION_OP_ADD(a,b) (a + b)
#define NANOS_REDUCTION_OP_SUB(a,b) (a - b)
#define NANOS_REDUCTION_OP_PROD(a,b) (a * b)
#define NANOS_REDUCTION_OP_AND(a,b) (a & b)
#define NANOS_REDUCTION_OP_OR(a,b) (a | b)
#define NANOS_REDUCTION_OP_XOR(a,b) (a ^ b)
#define NANOS_REDUCTION_OP_LAND(a,b) (a && b)
#define NANOS_REDUCTION_OP_LOR(a,b) (a || b)
#define NANOS_REDUCTION_OP_MAX(a,b) ((a > b)? a : b)
#define NANOS_REDUCTION_OP_MIN(a,b) ((a < b)? a : b)

#define NANOS_REDUCTION_DECL(Op,Type)\
   void nanos_reduction_bop_##Op##_##Type ( void *arg1, void *arg2); \
   void nanos_reduction_bop_##Op##_##Type##_ ( void *arg1, void *arg2); \
   void nanos_reduction_vop_##Op##_##Type ( int i, void *arg1, void *arg2); \
   void nanos_reduction_vop_##Op##_##Type##_ ( int i, void *arg1, void *arg2); \

#define NANOS_STRINGIZE(X) #X

#define NANOS_REDUCTION_DEF(Op,Op2,Type,Type2) \
   void nanos_reduction_bop_##Op##_##Type ( void *arg1, void *arg2) \
   { \
      Type2 *s = (Type2 *) arg1; \
      Type2 *v = (Type2 *) arg2; \
      *s = Op2(*s,*v); \
   } \
   __attribute__((alias(NANOS_STRINGIZE(nanos_reduction_bop_##Op## _##Type)))) \
   void nanos_reduction_bop_##Op##_##Type##_ ( void *arg1, void *arg2); \
   void nanos_reduction_vop_##Op##_##Type ( int n, void *arg1, void *arg2) \
   { \
      int i; \
      Type2 *s = (Type2 *) arg1; \
      Type2 *v = (Type2 *) arg2; \
      for (i = 0; i < n; i++) *s = Op2(*s,v[i]); \
   } \
   __attribute__((alias(NANOS_STRINGIZE(nanos_reduction_bop_##Op## _##Type)))) \
   void nanos_reduction_vop_##Op##_##Type##_ ( int n, void *arg1, void *arg2); \

#define NANOS_REDUCTION_COMPLEX_DEF(Op,Op2,Type,Type2) \
   void nanos_reduction_bop_##Op##_##Type ( void *arg1, void *arg2) \
   { \
      Type2 *s = (Type2 *) arg1; \
      Type2 *v = (Type2 *) arg2; \
      *s = Op2(__real__ *s, __real__ *v); \
   } \
   __attribute__((alias(NANOS_STRINGIZE(nanos_reduction_bop_##Op## _##Type)))) \
   void nanos_reduction_bop_##Op##_##Type##_ ( void *arg1, void *arg2); \
   void nanos_reduction_vop_##Op##_##Type ( int n, void *arg1, void *arg2) \
   { \
      int i; \
      Type2 *s = (Type2 *) arg1; \
      Type2 *v = (Type2 *) arg2; \
      for (i = 0; i < n; i++) *s = Op2(__real__ *s, __real__ v[i]); \
   } \
   __attribute__((alias(NANOS_STRINGIZE(nanos_reduction_bop_##Op## _##Type)))) \
   void nanos_reduction_vop_##Op##_##Type##_ ( int n, void *arg1, void *arg2); \

#define NANOS_REDUCTION_INT_TYPES_DECL(Op) \
   NANOS_REDUCTION_DECL(Op,char) \
   NANOS_REDUCTION_DECL(Op,uchar) \
   NANOS_REDUCTION_DECL(Op,schar) \
   NANOS_REDUCTION_DECL(Op,short) \
   NANOS_REDUCTION_DECL(Op,ushort) \
   NANOS_REDUCTION_DECL(Op,int) \
   NANOS_REDUCTION_DECL(Op,uint) \
   NANOS_REDUCTION_DECL(Op,long) \
   NANOS_REDUCTION_DECL(Op,ulong) \
   NANOS_REDUCTION_DECL(Op,longlong) \
   NANOS_REDUCTION_DECL(Op,ulonglong) \
   NANOS_REDUCTION_DECL(Op,_Bool) 

#define NANOS_REDUCTION_REAL_TYPES_DECL(Op) \
   NANOS_REDUCTION_DECL(Op,float) \
   NANOS_REDUCTION_DECL(Op,double) \
   NANOS_REDUCTION_DECL(Op,longdouble)

#define NANOS_REDUCTION_COMPLEX_TYPES_DECL(Op) \
   NANOS_REDUCTION_DECL(Op,cfloat) \
   NANOS_REDUCTION_DECL(Op,cdouble) \
   NANOS_REDUCTION_DECL(Op,clongdouble)

#define NANOS_REDUCTION_INT_TYPES_DEF(Op,Op2) \
   NANOS_REDUCTION_DEF(Op,Op2,char,char) \
   NANOS_REDUCTION_DEF(Op,Op2,uchar,unsigned char) \
   NANOS_REDUCTION_DEF(Op,Op2,schar,signed char) \
   NANOS_REDUCTION_DEF(Op,Op2,short,short) \
   NANOS_REDUCTION_DEF(Op,Op2,ushort,unsigned short) \
   NANOS_REDUCTION_DEF(Op,Op2,int,int) \
   NANOS_REDUCTION_DEF(Op,Op2,uint,unsigned int) \
   NANOS_REDUCTION_DEF(Op,Op2,long,long) \
   NANOS_REDUCTION_DEF(Op,Op2,ulong,unsigned long) \
   NANOS_REDUCTION_DEF(Op,Op2,longlong,long long) \
   NANOS_REDUCTION_DEF(Op,Op2,ulonglong,unsigned long long)

#define NANOS_REDUCTION_REAL_TYPES_DEF(Op,Op2) \
   NANOS_REDUCTION_DEF(Op,Op2,float,float) \
   NANOS_REDUCTION_DEF(Op,Op2,double,double) \
   NANOS_REDUCTION_DEF(Op,Op2,longdouble,long double)

#define NANOS_REDUCTION_COMPLEX_TYPES_DEF(Op,Op2) \
   NANOS_REDUCTION_COMPLEX_DEF(Op,Op2,cfloat,_Complex float) \
   NANOS_REDUCTION_COMPLEX_DEF(Op,Op2,cdouble,_Complex double) \
   NANOS_REDUCTION_COMPLEX_DEF(Op,Op2,clongdouble,_Complex long double)

// REDUCTION BUILTIN DECLARATION
NANOS_REDUCTION_INT_TYPES_DECL(add)
NANOS_REDUCTION_REAL_TYPES_DECL(add)
NANOS_REDUCTION_COMPLEX_TYPES_DECL(add)

NANOS_REDUCTION_INT_TYPES_DECL(sub)
NANOS_REDUCTION_REAL_TYPES_DECL(sub)
NANOS_REDUCTION_COMPLEX_TYPES_DECL(sub)

NANOS_REDUCTION_INT_TYPES_DECL(prod)
NANOS_REDUCTION_REAL_TYPES_DECL(prod)
NANOS_REDUCTION_COMPLEX_TYPES_DECL(prod)

NANOS_REDUCTION_INT_TYPES_DECL(and)

NANOS_REDUCTION_INT_TYPES_DECL(or)

NANOS_REDUCTION_INT_TYPES_DECL(xor)

NANOS_REDUCTION_INT_TYPES_DECL(land)
NANOS_REDUCTION_REAL_TYPES_DECL(land)
NANOS_REDUCTION_COMPLEX_TYPES_DECL(land)

NANOS_REDUCTION_INT_TYPES_DECL(lor)
NANOS_REDUCTION_REAL_TYPES_DECL(lor)
NANOS_REDUCTION_COMPLEX_TYPES_DECL(lor)

NANOS_REDUCTION_INT_TYPES_DECL(max)
NANOS_REDUCTION_REAL_TYPES_DECL(max)

NANOS_REDUCTION_INT_TYPES_DECL(min)
NANOS_REDUCTION_REAL_TYPES_DECL(min)


#define NANOS_REDUCTION_CLEANUP_DECL(Op, Type) \
   void nanos_reduction_default_cleanup_##Op ( void *r);

#define NANOS_REDUCTION_CLEANUP_DEF(Op, Type) \
   void nanos_reduction_default_cleanup_##Op ( void *r ) \
   { \
      nanos_reduction_t *red = (nanos_reduction_t *) r; \
      delete[] (Type *) red->privates; \
   }

NANOS_REDUCTION_CLEANUP_DECL(char, char)
NANOS_REDUCTION_CLEANUP_DECL(uchar, unsigned char)
NANOS_REDUCTION_CLEANUP_DECL(schar, signed char)
NANOS_REDUCTION_CLEANUP_DECL(short, short)
NANOS_REDUCTION_CLEANUP_DECL(ushort, unsigned short)
NANOS_REDUCTION_CLEANUP_DECL(int, int)
NANOS_REDUCTION_CLEANUP_DECL(uint, unsigned int)
NANOS_REDUCTION_CLEANUP_DECL(long, long)
NANOS_REDUCTION_CLEANUP_DECL(ulong, unsigned long)
NANOS_REDUCTION_CLEANUP_DECL(longlong, long long)
NANOS_REDUCTION_CLEANUP_DECL(ulonglong, unsigned long long )
NANOS_REDUCTION_CLEANUP_DECL(_Bool, _Bool)
NANOS_REDUCTION_CLEANUP_DECL(float, float)
NANOS_REDUCTION_CLEANUP_DECL(double, double)
NANOS_REDUCTION_CLEANUP_DECL(longdouble, long double)
NANOS_REDUCTION_CLEANUP_DECL(cfloat, _Complex float)
NANOS_REDUCTION_CLEANUP_DECL(cdouble, _Complex double)
NANOS_REDUCTION_CLEANUP_DECL(clongdouble, _Complex long double)

NANOS_API_DECL(void, nanos_reduction_default_cleanup_fortran, (void*));

#ifdef __cplusplus
}
#endif

#endif

