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

/*************************************************************************************/
/* AUTOMATIC GENERATED TEST                                                          */
/*                                                                                   */
/* Check %(srcdir)/tests/mcc/omp in order to get mercurium source code               */
/*************************************************************************************/

/*
<testinfo>
  test_generator=gens/api-omp-generator
</testinfo>
*/

#include <stdint.h>

typedef unsigned long int size_t;
struct  cudaPitchedPtr
{
  void *ptr;
  size_t pitch;
  size_t xsize;
  size_t ysize;
};
static __inline  struct cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz)
{
  struct cudaPitchedPtr s;
  s.ptr = d;
  s.pitch = p;
  s.xsize = xsz;
  s.ysize = ysz;
  return s;
}
struct  cudaPos
{
  size_t x;
  size_t y;
  size_t z;
};
static __inline  struct cudaPos make_cudaPos(size_t x, size_t y, size_t z)
{
  struct cudaPos p;
  p.x = x;
  p.y = y;
  p.z = z;
  return p;
}
struct  cudaExtent
{
  size_t width;
  size_t height;
  size_t depth;
};
static __inline  struct cudaExtent make_cudaExtent(size_t w, size_t h, size_t d)
{
  struct cudaExtent e;
  e.width = w;
  e.height = h;
  e.depth = d;
  return e;
}
struct  char1
{
  signed char x;
};
typedef struct char1 char1;
static __inline  char1 make_char1(signed char x)
{
  char1 t;
  t.x = x;
  return t;
}
struct  uchar1
{
  unsigned char x;
};
typedef struct uchar1 uchar1;
static __inline  uchar1 make_uchar1(unsigned char x)
{
  uchar1 t;
  t.x = x;
  return t;
}
struct __attribute__((aligned(2)))  char2
{
  signed char x;
  signed char y;
};
typedef struct char2 char2;
static __inline  char2 make_char2(signed char x, signed char y)
{
  char2 t;
  t.x = x;
  t.y = y;
  return t;
}
struct __attribute__((aligned(2)))  uchar2
{
  unsigned char x;
  unsigned char y;
};
typedef struct uchar2 uchar2;
static __inline  uchar2 make_uchar2(unsigned char x, unsigned char y)
{
  uchar2 t;
  t.x = x;
  t.y = y;
  return t;
}
struct  char3
{
  signed char x;
  signed char y;
  signed char z;
};
typedef struct char3 char3;
static __inline  char3 make_char3(signed char x, signed char y, signed char z)
{
  char3 t;
  t.x = x;
  t.y = y;
  t.z = z;
  return t;
}
struct  uchar3
{
  unsigned char x;
  unsigned char y;
  unsigned char z;
};
typedef struct uchar3 uchar3;
static __inline  uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z)
{
  uchar3 t;
  t.x = x;
  t.y = y;
  t.z = z;
  return t;
}
struct __attribute__((aligned(4)))  char4
{
  signed char x;
  signed char y;
  signed char z;
  signed char w;
};
typedef struct char4 char4;
static __inline  char4 make_char4(signed char x, signed char y, signed char z, signed char w)
{
  char4 t;
  t.x = x;
  t.y = y;
  t.z = z;
  t.w = w;
  return t;
}
struct __attribute__((aligned(4)))  uchar4
{
  unsigned char x;
  unsigned char y;
  unsigned char z;
  unsigned char w;
};
typedef struct uchar4 uchar4;
static __inline  uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w)
{
  uchar4 t;
  t.x = x;
  t.y = y;
  t.z = z;
  t.w = w;
  return t;
}
struct  short1
{
  short int x;
};
typedef struct short1 short1;
static __inline  short1 make_short1(short int x)
{
  short1 t;
  t.x = x;
  return t;
}
struct  ushort1
{
  unsigned short int x;
};
typedef struct ushort1 ushort1;
static __inline  ushort1 make_ushort1(unsigned short int x)
{
  ushort1 t;
  t.x = x;
  return t;
}
struct __attribute__((aligned(4)))  short2
{
  short int x;
  short int y;
};
typedef struct short2 short2;
static __inline  short2 make_short2(short int x, short int y)
{
  short2 t;
  t.x = x;
  t.y = y;
  return t;
}
struct __attribute__((aligned(4)))  ushort2
{
  unsigned short int x;
  unsigned short int y;
};
typedef struct ushort2 ushort2;
static __inline  ushort2 make_ushort2(unsigned short int x, unsigned short int y)
{
  ushort2 t;
  t.x = x;
  t.y = y;
  return t;
}
struct  short3
{
  short int x;
  short int y;
  short int z;
};
typedef struct short3 short3;
static __inline  short3 make_short3(short int x, short int y, short int z)
{
  short3 t;
  t.x = x;
  t.y = y;
  t.z = z;
  return t;
}
struct  ushort3
{
  unsigned short int x;
  unsigned short int y;
  unsigned short int z;
};
typedef struct ushort3 ushort3;
static __inline  ushort3 make_ushort3(unsigned short int x, unsigned short int y, unsigned short int z)
{
  ushort3 t;
  t.x = x;
  t.y = y;
  t.z = z;
  return t;
}
struct __attribute__((aligned(8)))  short4
{
  short int x;
  short int y;
  short int z;
  short int w;
};
typedef struct short4 short4;
static __inline  short4 make_short4(short int x, short int y, short int z, short int w)
{
  short4 t;
  t.x = x;
  t.y = y;
  t.z = z;
  t.w = w;
  return t;
}
struct __attribute__((aligned(8)))  ushort4
{
  unsigned short int x;
  unsigned short int y;
  unsigned short int z;
  unsigned short int w;
};
typedef struct ushort4 ushort4;
static __inline  ushort4 make_ushort4(unsigned short int x, unsigned short int y, unsigned short int z, unsigned short int w)
{
  ushort4 t;
  t.x = x;
  t.y = y;
  t.z = z;
  t.w = w;
  return t;
}
struct  int1
{
  int x;
};
typedef struct int1 int1;
static __inline  int1 make_int1(int x)
{
  int1 t;
  t.x = x;
  return t;
}
struct  uint1
{
  unsigned int x;
};
typedef struct uint1 uint1;
static __inline  uint1 make_uint1(unsigned int x)
{
  uint1 t;
  t.x = x;
  return t;
}
struct __attribute__((aligned(8)))  int2
{
  int x;
  int y;
};
typedef struct int2 int2;
static __inline  int2 make_int2(int x, int y)
{
  int2 t;
  t.x = x;
  t.y = y;
  return t;
}
struct __attribute__((aligned(8)))  uint2
{
  unsigned int x;
  unsigned int y;
};
typedef struct uint2 uint2;
static __inline  uint2 make_uint2(unsigned int x, unsigned int y)
{
  uint2 t;
  t.x = x;
  t.y = y;
  return t;
}
struct  int3
{
  int x;
  int y;
  int z;
};
typedef struct int3 int3;
static __inline  int3 make_int3(int x, int y, int z)
{
  int3 t;
  t.x = x;
  t.y = y;
  t.z = z;
  return t;
}
struct  uint3
{
  unsigned int x;
  unsigned int y;
  unsigned int z;
};
typedef struct uint3 uint3;
static __inline  uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z)
{
  uint3 t;
  t.x = x;
  t.y = y;
  t.z = z;
  return t;
}
struct __attribute__((aligned(16)))  int4
{
  int x;
  int y;
  int z;
  int w;
};
typedef struct int4 int4;
static __inline  int4 make_int4(int x, int y, int z, int w)
{
  int4 t;
  t.x = x;
  t.y = y;
  t.z = z;
  t.w = w;
  return t;
}
struct __attribute__((aligned(16)))  uint4
{
  unsigned int x;
  unsigned int y;
  unsigned int z;
  unsigned int w;
};
typedef struct uint4 uint4;
static __inline  uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
{
  uint4 t;
  t.x = x;
  t.y = y;
  t.z = z;
  t.w = w;
  return t;
}
struct  long1
{
  long int x;
};
typedef struct long1 long1;
static __inline  long1 make_long1(long int x)
{
  long1 t;
  t.x = x;
  return t;
}
struct  ulong1
{
  unsigned long int x;
};
typedef struct ulong1 ulong1;
static __inline  ulong1 make_ulong1(unsigned long int x)
{
  ulong1 t;
  t.x = x;
  return t;
}
struct __attribute__((aligned(2 * sizeof(long int))))  long2
{
  long int x;
  long int y;
};
typedef struct long2 long2;
static __inline  long2 make_long2(long int x, long int y)
{
  long2 t;
  t.x = x;
  t.y = y;
  return t;
}
struct __attribute__((aligned(2 * sizeof(unsigned long int))))  ulong2
{
  unsigned long int x;
  unsigned long int y;
};
typedef struct ulong2 ulong2;
static __inline  ulong2 make_ulong2(unsigned long int x, unsigned long int y)
{
  ulong2 t;
  t.x = x;
  t.y = y;
  return t;
}
struct  long3
{
  long int x;
  long int y;
  long int z;
};
typedef struct long3 long3;
static __inline  long3 make_long3(long int x, long int y, long int z)
{
  long3 t;
  t.x = x;
  t.y = y;
  t.z = z;
  return t;
}
struct  ulong3
{
  unsigned long int x;
  unsigned long int y;
  unsigned long int z;
};
typedef struct ulong3 ulong3;
static __inline  ulong3 make_ulong3(unsigned long int x, unsigned long int y, unsigned long int z)
{
  ulong3 t;
  t.x = x;
  t.y = y;
  t.z = z;
  return t;
}
struct __attribute__((aligned(16)))  long4
{
  long int x;
  long int y;
  long int z;
  long int w;
};
typedef struct long4 long4;
static __inline  long4 make_long4(long int x, long int y, long int z, long int w)
{
  long4 t;
  t.x = x;
  t.y = y;
  t.z = z;
  t.w = w;
  return t;
}
struct __attribute__((aligned(16)))  ulong4
{
  unsigned long int x;
  unsigned long int y;
  unsigned long int z;
  unsigned long int w;
};
typedef struct ulong4 ulong4;
static __inline  ulong4 make_ulong4(unsigned long int x, unsigned long int y, unsigned long int z, unsigned long int w)
{
  ulong4 t;
  t.x = x;
  t.y = y;
  t.z = z;
  t.w = w;
  return t;
}
struct  float1
{
  float x;
};
typedef struct float1 float1;
static __inline  float1 make_float1(float x)
{
  float1 t;
  t.x = x;
  return t;
}
struct __attribute__((aligned(8)))  float2
{
  float x;
  float y;
};
typedef struct float2 float2;
static __inline  float2 make_float2(float x, float y)
{
  float2 t;
  t.x = x;
  t.y = y;
  return t;
}
struct  float3
{
  float x;
  float y;
  float z;
};
typedef struct float3 float3;
static __inline  float3 make_float3(float x, float y, float z)
{
  float3 t;
  t.x = x;
  t.y = y;
  t.z = z;
  return t;
}
struct __attribute__((aligned(16)))  float4
{
  float x;
  float y;
  float z;
  float w;
};
typedef struct float4 float4;
static __inline  float4 make_float4(float x, float y, float z, float w)
{
  float4 t;
  t.x = x;
  t.y = y;
  t.z = z;
  t.w = w;
  return t;
}
struct  longlong1
{
  long long int x;
};
typedef struct longlong1 longlong1;
static __inline  longlong1 make_longlong1(long long int x)
{
  longlong1 t;
  t.x = x;
  return t;
}
struct  ulonglong1
{
  unsigned long long int x;
};
typedef struct ulonglong1 ulonglong1;
static __inline  ulonglong1 make_ulonglong1(unsigned long long int x)
{
  ulonglong1 t;
  t.x = x;
  return t;
}
struct __attribute__((aligned(16)))  longlong2
{
  long long int x;
  long long int y;
};
typedef struct longlong2 longlong2;
static __inline  longlong2 make_longlong2(long long int x, long long int y)
{
  longlong2 t;
  t.x = x;
  t.y = y;
  return t;
}
struct __attribute__((aligned(16)))  ulonglong2
{
  unsigned long long int x;
  unsigned long long int y;
};
typedef struct ulonglong2 ulonglong2;
static __inline  ulonglong2 make_ulonglong2(unsigned long long int x, unsigned long long int y)
{
  ulonglong2 t;
  t.x = x;
  t.y = y;
  return t;
}
struct  longlong3
{
  long long int x;
  long long int y;
  long long int z;
};
typedef struct longlong3 longlong3;
static __inline  longlong3 make_longlong3(long long int x, long long int y, long long int z)
{
  longlong3 t;
  t.x = x;
  t.y = y;
  t.z = z;
  return t;
}
struct  ulonglong3
{
  unsigned long long int x;
  unsigned long long int y;
  unsigned long long int z;
};
typedef struct ulonglong3 ulonglong3;
static __inline  ulonglong3 make_ulonglong3(unsigned long long int x, unsigned long long int y, unsigned long long int z)
{
  ulonglong3 t;
  t.x = x;
  t.y = y;
  t.z = z;
  return t;
}
struct __attribute__((aligned(16)))  longlong4
{
  long long int x;
  long long int y;
  long long int z;
  long long int w;
};
typedef struct longlong4 longlong4;
static __inline  longlong4 make_longlong4(long long int x, long long int y, long long int z, long long int w)
{
  longlong4 t;
  t.x = x;
  t.y = y;
  t.z = z;
  t.w = w;
  return t;
}
struct __attribute__((aligned(16)))  ulonglong4
{
  unsigned long long int x;
  unsigned long long int y;
  unsigned long long int z;
  unsigned long long int w;
};
typedef struct ulonglong4 ulonglong4;
static __inline  ulonglong4 make_ulonglong4(unsigned long long int x, unsigned long long int y, unsigned long long int z, unsigned long long int w)
{
  ulonglong4 t;
  t.x = x;
  t.y = y;
  t.z = z;
  t.w = w;
  return t;
}
struct  double1
{
  double x;
};
typedef struct double1 double1;
static __inline  double1 make_double1(double x)
{
  double1 t;
  t.x = x;
  return t;
}
struct __attribute__((aligned(16)))  double2
{
  double x;
  double y;
};
typedef struct double2 double2;
static __inline  double2 make_double2(double x, double y)
{
  double2 t;
  t.x = x;
  t.y = y;
  return t;
}
struct  double3
{
  double x;
  double y;
  double z;
};
typedef struct double3 double3;
static __inline  double3 make_double3(double x, double y, double z)
{
  double3 t;
  t.x = x;
  t.y = y;
  t.z = z;
  return t;
}
struct __attribute__((aligned(16)))  double4
{
  double x;
  double y;
  double z;
  double w;
};
typedef struct double4 double4;
static __inline  double4 make_double4(double x, double y, double z, double w)
{
  double4 t;
  t.x = x;
  t.y = y;
  t.z = z;
  t.w = w;
  return t;
}
extern int signgam;
enum mcc_enum_anon_7
{
  _IEEE_ =  -1,
  _SVID_ = 0,
  _XOPEN_ = 1,
  _POSIX_ = 2,
  _ISOC_ = 3
};
typedef enum mcc_enum_anon_7 _LIB_VERSION_TYPE;
extern _LIB_VERSION_TYPE _LIB_VERSION;
typedef float2 cuFloatComplex;
static __inline  float cuCrealf(cuFloatComplex x)
{
  return x.x;
}
static __inline  float cuCimagf(cuFloatComplex x)
{
  return x.y;
}
static __inline  cuFloatComplex make_cuFloatComplex(float r, float i)
{
  cuFloatComplex res;
  res.x = r;
  res.y = i;
  return res;
}
static __inline  cuFloatComplex cuConjf(cuFloatComplex x)
{
  return make_cuFloatComplex(cuCrealf(x),  -cuCimagf(x));
}
static __inline  cuFloatComplex cuCaddf(cuFloatComplex x, cuFloatComplex y)
{
  return make_cuFloatComplex(cuCrealf(x) + cuCrealf(y), cuCimagf(x) + cuCimagf(y));
}
static __inline  cuFloatComplex cuCsubf(cuFloatComplex x, cuFloatComplex y)
{
  return make_cuFloatComplex(cuCrealf(x) - cuCrealf(y), cuCimagf(x) - cuCimagf(y));
}
static __inline  cuFloatComplex cuCmulf(cuFloatComplex x, cuFloatComplex y)
{
  cuFloatComplex prod;
  prod = make_cuFloatComplex(cuCrealf(x) * cuCrealf(y) - cuCimagf(x) * cuCimagf(y), cuCrealf(x) * cuCimagf(y) + cuCimagf(x) * cuCrealf(y));
  return prod;
}
extern float fabsf(float __x) __attribute__((__nothrow__)) __attribute__((__const__));
static __inline  cuFloatComplex cuCdivf(cuFloatComplex x, cuFloatComplex y)
{
  cuFloatComplex quot;
  float s = fabsf(cuCrealf(y)) + fabsf(cuCimagf(y));
  float oos = 1.000000000000000000000000e+00f / s;
  float ars = cuCrealf(x) * oos;
  float ais = cuCimagf(x) * oos;
  float brs = cuCrealf(y) * oos;
  float bis = cuCimagf(y) * oos;
  s = brs * brs + bis * bis;
  oos = 1.000000000000000000000000e+00f / s;
  quot = make_cuFloatComplex((ars * brs + ais * bis) * oos, (ais * brs - ars * bis) * oos);
  return quot;
}
extern float sqrtf(float __x) __attribute__((__nothrow__));
static __inline  float cuCabsf(cuFloatComplex x)
{
  float v;
  float w;
  float t;
  float a = cuCrealf(x);
  float b = cuCimagf(x);
  a = fabsf(a);
  b = fabsf(b);
  if (a > b)
    {
      v = a;
      w = b;
    }
  else
    {
      v = b;
      w = a;
    }
  t = w / v;
  t = 1.000000000000000000000000e+00f + t * t;
  t = v * sqrtf(t);
  if ((v == 0.000000000000000000000000e+00f || v > 3.402823466385288598117042e+38f) || w > 3.402823466385288598117042e+38f)
    {
      t = v + w;
    }
  return t;
}
typedef double2 cuDoubleComplex;
static __inline  double cuCreal(cuDoubleComplex x)
{
  return x.x;
}
static __inline  double cuCimag(cuDoubleComplex x)
{
  return x.y;
}
static __inline  cuDoubleComplex make_cuDoubleComplex(double r, double i)
{
  cuDoubleComplex res;
  res.x = r;
  res.y = i;
  return res;
}
static __inline  cuDoubleComplex cuConj(cuDoubleComplex x)
{
  return make_cuDoubleComplex(cuCreal(x),  -cuCimag(x));
}
static __inline  cuDoubleComplex cuCadd(cuDoubleComplex x, cuDoubleComplex y)
{
  return make_cuDoubleComplex(cuCreal(x) + cuCreal(y), cuCimag(x) + cuCimag(y));
}
static __inline  cuDoubleComplex cuCsub(cuDoubleComplex x, cuDoubleComplex y)
{
  return make_cuDoubleComplex(cuCreal(x) - cuCreal(y), cuCimag(x) - cuCimag(y));
}
static __inline  cuDoubleComplex cuCmul(cuDoubleComplex x, cuDoubleComplex y)
{
  cuDoubleComplex prod;
  prod = make_cuDoubleComplex(cuCreal(x) * cuCreal(y) - cuCimag(x) * cuCimag(y), cuCreal(x) * cuCimag(y) + cuCimag(x) * cuCreal(y));
  return prod;
}
extern double fabs(double __x) __attribute__((__nothrow__)) __attribute__((__const__));
static __inline  cuDoubleComplex cuCdiv(cuDoubleComplex x, cuDoubleComplex y)
{
  cuDoubleComplex quot;
  double s = fabs(cuCreal(y)) + fabs(cuCimag(y));
  double oos = 1.00000000000000000000000000000000000000000000000000000e+00 / s;
  double ars = cuCreal(x) * oos;
  double ais = cuCimag(x) * oos;
  double brs = cuCreal(y) * oos;
  double bis = cuCimag(y) * oos;
  s = brs * brs + bis * bis;
  oos = 1.00000000000000000000000000000000000000000000000000000e+00 / s;
  quot = make_cuDoubleComplex((ars * brs + ais * bis) * oos, (ais * brs - ars * bis) * oos);
  return quot;
}
extern double sqrt(double __x) __attribute__((__nothrow__));
static __inline  double cuCabs(cuDoubleComplex x)
{
  double v;
  double w;
  double t;
  double a = cuCreal(x);
  double b = cuCimag(x);
  a = fabs(a);
  b = fabs(b);
  if (a > b)
    {
      v = a;
      w = b;
    }
  else
    {
      v = b;
      w = a;
    }
  t = w / v;
  t = 1.00000000000000000000000000000000000000000000000000000e+00 + t * t;
  t = v * sqrt(t);
  if ((v == 0.00000000000000000000000000000000000000000000000000000e+00 || v > 1.79769313486231570814527423731704356798070567525844997e+308) || w > 1.79769313486231570814527423731704356798070567525844997e+308)
    {
      t = v + w;
    }
  return t;
}
typedef cuFloatComplex cuComplex;
static __inline  cuComplex make_cuComplex(float x, float y)
{
  return make_cuFloatComplex(x, y);
}
static __inline  cuDoubleComplex cuComplexFloatToDouble(cuFloatComplex c)
{
  return make_cuDoubleComplex((double)cuCrealf(c), (double)cuCimagf(c));
}
static __inline  cuFloatComplex cuComplexDoubleToFloat(cuDoubleComplex c)
{
  return make_cuFloatComplex((float)cuCreal(c), (float)cuCimag(c));
}
static __inline  cuComplex cuCfmaf(cuComplex x, cuComplex y, cuComplex d)
{
  float real_res;
  float imag_res;
  real_res = cuCrealf(x) * cuCrealf(y) + cuCrealf(d);
  imag_res = cuCrealf(x) * cuCimagf(y) + cuCimagf(d);
  real_res =  -(cuCimagf(x) * cuCimagf(y)) + real_res;
  imag_res = cuCimagf(x) * cuCrealf(y) + imag_res;
  return make_cuComplex(real_res, imag_res);
}
static __inline  cuDoubleComplex cuCfma(cuDoubleComplex x, cuDoubleComplex y, cuDoubleComplex d)
{
  double real_res;
  double imag_res;
  real_res = cuCreal(x) * cuCreal(y) + cuCreal(d);
  imag_res = cuCreal(x) * cuCimag(y) + cuCimag(d);
  real_res =  -(cuCimag(x) * cuCimag(y)) + real_res;
  imag_res = cuCimag(x) * cuCreal(y) + imag_res;
  return make_cuDoubleComplex(real_res, imag_res);
}
struct _IO_FILE_plus;
extern struct _IO_FILE_plus _IO_2_1_stdin_;
extern struct _IO_FILE_plus _IO_2_1_stdout_;
extern struct _IO_FILE_plus _IO_2_1_stderr_;
struct _IO_FILE;
extern struct _IO_FILE *stdin;
extern struct _IO_FILE *stdout;
extern struct _IO_FILE *stderr;
extern int sys_nerr;
extern const char *const sys_errlist[];
struct  mcc_struct_anon_15
{
  void (*outline)(void *);
};
typedef struct mcc_struct_anon_15 nanos_smp_args_t;
struct  nanos_args_1_t
{
  int *i;
  int (*a)[16];
};
static void smp_ol_main_1(struct nanos_args_1_t *const args);
struct  mcc_struct_anon_11
{
  _Bool mandatory_creation:1;
  _Bool tied:1;
  _Bool clear_chunk:1;
  _Bool reserved0:1;
  _Bool reserved1:1;
  _Bool reserved2:1;
  _Bool reserved3:1;
  _Bool reserved4:1;
};
typedef struct mcc_struct_anon_11 nanos_wd_props_t;
struct  nanos_const_wd_definition_tag
{
  nanos_wd_props_t props;
  size_t data_alignment;
  size_t num_copies;
  size_t num_devices;
  size_t num_dimensions;
  const char *description;
};
typedef struct nanos_const_wd_definition_tag nanos_const_wd_definition_t;
struct  mcc_struct_anon_14
{
  void *(*factory)(void *);
  void *arg;
};
typedef struct mcc_struct_anon_14 nanos_device_t;
struct  nanos_const_wd_definition_1
{
  nanos_const_wd_definition_t base;
  nanos_device_t devices[1];
};
extern void *nanos_smp_factory(void *args);
extern int nanos_omp_get_num_threads_next_parallel(int threads_requested);
typedef void *nanos_team_t;
typedef void *nanos_thread_t;
enum mcc_enum_anon_5
{
  NANOS_OK = 0,
  NANOS_UNKNOWN_ERR = 1,
  NANOS_UNIMPLEMENTED = 2,
  NANOS_ENOMEM = 3,
  NANOS_INVALID_PARAM = 4,
  NANOS_INVALID_REQUEST = 5
};
typedef enum mcc_enum_anon_5 nanos_err_t;
struct mcc_struct_anon_18;
typedef struct mcc_struct_anon_18 nanos_constraint_t;
typedef void *nanos_sched_t;
extern nanos_err_t nanos_create_team(nanos_team_t *team, nanos_sched_t sg, unsigned int *nthreads, nanos_constraint_t *constraints, _Bool reuse, nanos_thread_t *info, nanos_const_wd_definition_t *const_data);
extern void nanos_handle_error(nanos_err_t err);
struct  mcc_struct_anon_12
{
  _Bool is_final:1;
  _Bool is_recover:1;
  _Bool is_implicit:1;
  _Bool reserved3:1;
  _Bool reserved4:1;
  _Bool reserved5:1;
  _Bool reserved6:1;
  _Bool reserved7:1;
};
typedef struct mcc_struct_anon_12 nanos_wd_dyn_flags_t;
struct  mcc_struct_anon_13
{
  nanos_wd_dyn_flags_t flags;
  nanos_thread_t tie_to;
  int priority;
};
typedef struct mcc_struct_anon_13 nanos_wd_dyn_props_t;
typedef void *nanos_wd_t;
struct mcc_struct_anon_4;
typedef struct mcc_struct_anon_4 nanos_copy_data_internal_t;
typedef nanos_copy_data_internal_t nanos_copy_data_t;
struct mcc_struct_anon_0;
typedef struct mcc_struct_anon_0 nanos_region_dimension_internal_t;
typedef void *nanos_wg_t;
extern nanos_err_t nanos_create_wd_compact(nanos_wd_t *wd, nanos_const_wd_definition_t *const_data, nanos_wd_dyn_props_t *dyn_props, size_t data_size, void **data, nanos_wg_t wg, nanos_copy_data_t **copies, nanos_region_dimension_internal_t **dimensions);
extern nanos_wd_t nanos_current_wd(void);
struct mcc_struct_anon_2;
typedef struct mcc_struct_anon_2 nanos_data_access_internal_t;
typedef nanos_data_access_internal_t nanos_data_access_t;
extern nanos_err_t nanos_submit(nanos_wd_t wd, size_t num_data_accesses, nanos_data_access_t *data_accesses, nanos_team_t team);
typedef void (*nanos_translate_args_t)(void *, nanos_wd_t);
extern nanos_err_t nanos_create_wd_and_run_compact(nanos_const_wd_definition_t *const_data, nanos_wd_dyn_props_t *dyn_props, size_t data_size, void *data, size_t num_data_accesses, nanos_data_access_t *data_accesses, nanos_copy_data_t *copies, nanos_region_dimension_internal_t *dimensions, nanos_translate_args_t translate_args);
struct  mcc_struct_anon_1
{
  _Bool input:1;
  _Bool output:1;
  _Bool can_rename:1;
  _Bool concurrent:1;
  _Bool commutative:1;
};
typedef struct mcc_struct_anon_1 nanos_access_type_internal_t;
typedef long int ptrdiff_t;
struct  mcc_struct_anon_2
{
  void *address;
  nanos_access_type_internal_t flags;
  short int dimension_count;
  const nanos_region_dimension_internal_t *dimensions;
  ptrdiff_t offset;
};
extern nanos_err_t nanos_end_team(nanos_team_t team);
typedef struct _IO_FILE FILE;
extern int fprintf(FILE *__restrict __stream, const char *__restrict __format, ...);
extern struct _IO_FILE *stderr;
int main(int argc, char **argv)
{
  int i;
  int a[16];
  int it;
  int rv = 0;
  for (i = 0; i < 16; i++)
    {
      a[i] = 0;
    }
  for (it = 0; it < 4; it++)
    {
      {
        nanos_err_t nanos_err;
        nanos_wd_dyn_props_t dyn_props;
        unsigned int nth_i;
        struct nanos_args_1_t imm_args;
        nanos_data_access_t dependences[1];
        static nanos_smp_args_t smp_ol_main_1_args = {.outline = (void (*)(void *))(void (*)(struct nanos_args_1_t *))&smp_ol_main_1};
        static struct nanos_const_wd_definition_1 nanos_wd_const_data = {.base = {.props = {.mandatory_creation = 1, .tied = 1, .clear_chunk = 0, .reserved0 = 0, .reserved1 = 0, .reserved2 = 0, .reserved3 = 0, .reserved4 = 0}, .data_alignment = __alignof__(struct nanos_args_1_t), .num_copies = 0, .num_devices = 1, .num_dimensions = 0, .description = 0}, .devices = {[0] = {.factory = &nanos_smp_factory, .arg = &smp_ol_main_1_args}}};
        unsigned int nanos_num_threads = nanos_omp_get_num_threads_next_parallel(0);
        nanos_team_t nanos_team = (void *)0;
        nanos_thread_t nanos_team_threads[nanos_num_threads];
        nanos_err = nanos_create_team(&nanos_team, (void *)0, &nanos_num_threads, (nanos_constraint_t *)0, 1, nanos_team_threads, &nanos_wd_const_data.base);
        if (nanos_err != NANOS_OK)
          {
            nanos_handle_error(nanos_err);
          }
        dyn_props.tie_to = (void *)0;
        dyn_props.priority = 0;
        dyn_props.flags.is_final = 0;
        for (nth_i = 1; nth_i < nanos_num_threads; nth_i = nth_i + 1)
          {
            dyn_props.tie_to = nanos_team_threads[nth_i];
            struct nanos_args_1_t *ol_args = 0;
            nanos_wd_t nanos_wd_ = (void *)0;
            nanos_err = nanos_create_wd_compact(&nanos_wd_, &nanos_wd_const_data.base, &dyn_props, sizeof(struct nanos_args_1_t), (void **)&ol_args, nanos_current_wd(), (nanos_copy_data_t **)0, (nanos_region_dimension_internal_t **)0);
            if (nanos_err != NANOS_OK)
              {
                nanos_handle_error(nanos_err);
              }
            (*ol_args).i = &i;
            (*ol_args).a = &a;
            nanos_err = nanos_submit(nanos_wd_, 0, (nanos_data_access_t *)0, (void *)0);
            if (nanos_err != NANOS_OK)
              {
                nanos_handle_error(nanos_err);
              }
          }
        dyn_props.tie_to = nanos_team_threads[0];
        imm_args.i = &i;
        imm_args.a = &a;
        nanos_err = nanos_create_wd_and_run_compact(&nanos_wd_const_data.base, &dyn_props, sizeof(struct nanos_args_1_t), &imm_args, 0, dependences, (nanos_copy_data_t *)0, (nanos_region_dimension_internal_t *)0, (void (*)(void *, nanos_wd_t))0);
        if (nanos_err != NANOS_OK)
          {
            nanos_handle_error(nanos_err);
          }
        nanos_err = nanos_end_team(nanos_team);
        if (nanos_err != NANOS_OK)
          {
            nanos_handle_error(nanos_err);
          }
      }
    }
  for (i = 0; i < 16; i++)
    {
      if (a[i] != 4)
        {
          rv++;
        }
    }
  if (rv)
    {
      fprintf(stderr, "%s: Error, final result is not valid \n", argv[0]);
    }
  return rv;
}
struct nanos_ws_desc;
typedef struct nanos_ws_desc nanos_ws_desc_t;
extern nanos_err_t nanos_omp_set_implicit(nanos_wd_t uwd);
typedef void *nanos_ws_item_t;
extern nanos_err_t nanos_worksharing_next_item(nanos_ws_desc_t *wsd, nanos_ws_item_t *wsi);
struct  mcc_struct_anon_10
{
  int64_t lower;
  int64_t upper;
  _Bool execute:1;
  _Bool last:1;
};
typedef struct mcc_struct_anon_10 nanos_ws_item_loop_t;
extern nanos_err_t nanos_omp_barrier(void);
static void smp_ol_main_0_unpacked(nanos_ws_desc_t *wsd_1, int *const a)
{
  int i;
  {
    {
      nanos_err_t nanos_err;
      nanos_err = nanos_omp_set_implicit(nanos_current_wd());
      if (nanos_err != NANOS_OK)
        {
          nanos_handle_error(nanos_err);
        }
    }
    {
      nanos_err_t nanos_err;
      nanos_ws_item_loop_t nanos_item_loop;
      nanos_err = nanos_worksharing_next_item(wsd_1, (void **)&nanos_item_loop);
      if (nanos_err != NANOS_OK)
        {
          nanos_handle_error(nanos_err);
        }
      while (nanos_item_loop.execute)
        {
          for (i = nanos_item_loop.lower; i <= nanos_item_loop.upper; i += 1)
            {
              {
                a[i]++;
              }
            }
          ;
          nanos_err = nanos_worksharing_next_item(wsd_1, (void **)&nanos_item_loop);
        }
    }
    {
      nanos_err_t nanos_err;
      nanos_err = nanos_omp_barrier();
      if (nanos_err != NANOS_OK)
        {
          nanos_handle_error(nanos_err);
        }
    }
  }
}
struct  nanos_args_0_t
{
  nanos_ws_desc_t *wsd_1;
  int (*a)[16];
};
static void smp_ol_main_0(struct nanos_args_0_t *const args)
{
  {
    smp_ol_main_0_unpacked((*args).wsd_1, *(*args).a);
  }
}
extern nanos_err_t nanos_enter_team(void);
typedef void *nanos_ws_t;
enum nanos_omp_sched_t
{
  nanos_omp_sched_static = 1,
  nanos_omp_sched_dynamic = 2,
  nanos_omp_sched_guided = 3,
  nanos_omp_sched_auto = 4
};
typedef enum nanos_omp_sched_t nanos_omp_sched_t;
extern nanos_ws_t nanos_omp_find_worksharing(nanos_omp_sched_t kind);
struct  mcc_struct_anon_9
{
  int64_t lower_bound;
  int64_t upper_bound;
  int64_t loop_step;
  int64_t chunk_size;
};
typedef struct mcc_struct_anon_9 nanos_ws_info_loop_t;
typedef void *nanos_ws_info_t;
extern nanos_err_t nanos_worksharing_create(nanos_ws_desc_t **wsd, nanos_ws_t ws, nanos_ws_info_t *info, _Bool *b);
extern nanos_err_t nanos_team_get_num_supporting_threads(int *n);
extern nanos_err_t nanos_malloc(void **p, size_t size, const char *file, int line);
typedef void *nanos_ws_data_t;
struct  nanos_ws_desc
{
  volatile nanos_ws_t ws;
  nanos_ws_data_t data;
  struct nanos_ws_desc *next;
  nanos_thread_t *threads;
  int nths;
};
extern nanos_err_t nanos_team_get_supporting_threads(int *n, nanos_thread_t *list_of_threads);
typedef void *nanos_slicer_t;
extern nanos_slicer_t nanos_find_slicer(const char *slicer);
extern nanos_err_t nanos_create_sliced_wd(nanos_wd_t *uwd, size_t num_devices, nanos_device_t *devices, size_t outline_data_size, int outline_data_align, void **outline_data, nanos_wg_t uwg, nanos_slicer_t slicer, nanos_wd_props_t *props, nanos_wd_dyn_props_t *dyn_props, size_t num_copies, nanos_copy_data_t **copies, size_t num_dimensions, nanos_region_dimension_internal_t **dimensions);
extern nanos_err_t nanos_free(void *p);
extern nanos_err_t nanos_leave_team(void);
static void smp_ol_main_1_unpacked(int *const i, int *const a)
{
  {
    nanos_err_t nanos_err;
    nanos_err = nanos_omp_set_implicit(nanos_current_wd());
    if (nanos_err != NANOS_OK)
      {
        nanos_handle_error(nanos_err);
      }
    nanos_err = nanos_enter_team();
    if (nanos_err != NANOS_OK)
      {
        nanos_handle_error(nanos_err);
      }
    {
      int nanos_chunk;
      nanos_ws_info_loop_t nanos_setup_info_loop;
      nanos_err_t nanos_err;
      nanos_ws_desc_t *wsd_1;
      _Bool single_guard;
      struct nanos_args_0_t imm_args;
      void *current_ws_policy = nanos_omp_find_worksharing(nanos_omp_sched_static);
      if (current_ws_policy == 0)
        {
          nanos_handle_error(NANOS_UNIMPLEMENTED);
        }
      nanos_chunk = 1;
      nanos_setup_info_loop.lower_bound = 0;
      nanos_setup_info_loop.upper_bound = 15;
      nanos_setup_info_loop.loop_step = 1;
      nanos_setup_info_loop.chunk_size = nanos_chunk;
      nanos_err = nanos_worksharing_create(&wsd_1, current_ws_policy, (void **)&nanos_setup_info_loop, &single_guard);
      if (nanos_err != NANOS_OK)
        {
          nanos_handle_error(nanos_err);
        }
      if (single_guard)
        {
          int sup_threads;
          nanos_err = nanos_team_get_num_supporting_threads(&sup_threads);
          if (nanos_err != NANOS_OK)
            {
              nanos_handle_error(nanos_err);
            }
          if (sup_threads > 0)
            {
              nanos_wd_dyn_props_t dyn_props;
              nanos_err = nanos_malloc((void **)&(*wsd_1).threads, sizeof(void *) * sup_threads, "", 0);
              if (nanos_err != NANOS_OK)
                {
                  nanos_handle_error(nanos_err);
                }
              nanos_err = nanos_team_get_supporting_threads(&(*wsd_1).nths, (*wsd_1).threads);
              if (nanos_err != NANOS_OK)
                {
                  nanos_handle_error(nanos_err);
                }
              struct nanos_args_0_t *ol_args = (struct nanos_args_0_t *)0;
              static nanos_smp_args_t smp_ol_main_0_args = {.outline = (void (*)(void *))(void (*)(struct nanos_args_0_t *))&smp_ol_main_0};
              static struct nanos_const_wd_definition_1 nanos_wd_const_data = {.base = {.props = {.mandatory_creation = 1, .tied = 1, .clear_chunk = 0, .reserved0 = 0, .reserved1 = 0, .reserved2 = 0, .reserved3 = 0, .reserved4 = 0}, .data_alignment = __alignof__(struct nanos_args_0_t), .num_copies = 0, .num_devices = 1, .num_dimensions = 0, .description = 0}, .devices = {[0] = {.factory = &nanos_smp_factory, .arg = &smp_ol_main_0_args}}};
              void *nanos_wd_ = (void *)0;
              dyn_props.tie_to = (void *)0;
              dyn_props.priority = 0;
              dyn_props.flags.is_final = 0;
              static void *replicate = (void *)0;
              if (replicate == (void *)0)
                {
                  replicate = nanos_find_slicer("replicate");
                }
              if (replicate == (void *)0)
                {
                  nanos_handle_error(NANOS_UNIMPLEMENTED);
                }
              nanos_err = nanos_create_sliced_wd(&nanos_wd_, nanos_wd_const_data.base.num_devices, nanos_wd_const_data.devices, (unsigned long int)sizeof(struct nanos_args_0_t), nanos_wd_const_data.base.data_alignment, (void **)&ol_args, (void **)0, replicate, &nanos_wd_const_data.base.props, &dyn_props, 0, (nanos_copy_data_t **)0, 0, (nanos_region_dimension_internal_t **)0);
              if (nanos_err != NANOS_OK)
                {
                  nanos_handle_error(nanos_err);
                }
              (*ol_args).wsd_1 = wsd_1;
              (*ol_args).a = (int (*)[16]) a;
              nanos_err = nanos_submit(nanos_wd_, 0, (nanos_data_access_t *)0, (void *)0);
              if (nanos_err != NANOS_OK)
                {
                  nanos_handle_error(nanos_err);
                }
              nanos_err = nanos_free((*wsd_1).threads);
              if (nanos_err != NANOS_OK)
                {
                  nanos_handle_error(nanos_err);
                }
            }
        }
      imm_args.wsd_1 = wsd_1;
      imm_args.a = (int (*)[16]) a;
      smp_ol_main_0(&(imm_args));
    }
    nanos_err = nanos_omp_barrier();
    if (nanos_err != NANOS_OK)
      {
        nanos_handle_error(nanos_err);
      }
    nanos_err = nanos_leave_team();
    if (nanos_err != NANOS_OK)
      {
        nanos_handle_error(nanos_err);
      }
  }
}
static void smp_ol_main_1(struct nanos_args_1_t *const args)
{
  {
    smp_ol_main_1_unpacked((*args).i, *(*args).a);
  }
}
