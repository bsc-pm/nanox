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

#ifndef _NANOS_COMPATIBILITY_HPP
#define _NANOS_COMPATIBILITY_HPP

// Define GCC Version
#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

// Define array and boundary pointers
#define NANOS_REGISTER(name, type, ...) \
    type __##name[] __attribute__((weak)) = {__VA_ARGS__}; \
    type * __##name##_begin __attribute__((weak)) = __##name; \
    type * __##name##_end __attribute__((weak)) = __##name + sizeof(__##name) / sizeof(*__##name);

// Keep compatibility with old LINKER_SECTION macro
#define LINKER_SECTION(name, type, nop) NANOS_REGISTER(name, type, nop)


#if __CUDACC__

#define BROKEN_COMPARE_AND_SWAP

#endif

#if __IBMCPP__
#define __volatile volatile // defining __volatile to volatile with XLC compiler. But you shouldn't have to do this.
#endif

// compiler issues

#if __GXX_EXPERIMENTAL_CXX0X__

#include <unordered_map>
#include <memory>

namespace TR1 = std;

#else

#include <tr1/unordered_map>
#include <tr1/memory>

namespace TR1 = std::tr1;

#endif

#ifdef __GNUC__
#if __GNUC__ == 4 && __GNUC_MINOR__ < 2

namespace std {
#ifndef __GXX_EXPERIMENTAL_CXX0X__
namespace tr1{
#endif // __GXX_EXPERIMENTAL_CXX0X__

/* Specialize hash for unsigned long long allows unordered_map<uint64_t, xxx> when compiling for 32 bits */
template<> struct hash<unsigned long long> : public std::unary_function<unsigned long long, std::size_t> { std::size_t operator()(unsigned long long val) const { return static_cast<std::size_t>(val); } };
}

#ifndef __GXX_EXPERIMENTAL_CXX0X__
}
#endif // __GXX_EXPERIMENTAL_CXX0X__

#endif // __GNUC__ == 4 && __GNUC_MINOR__ < 2
#endif // __GNUC__


#ifdef __GNUC__
// Dan Tsafrir [11/2/2011]: ugly hack to match the ugliness it fixes.
//
// Explanation:
//
// For the statements to which this macro is applied, gcc-4.1
//   (a) creates a temporary,
//   (b) copies it using the copy ctor to another temporary,
//   (c) invokes the operator=.
// But since the copy ctor in (b) does not exist => compile error.
// This macro prevents (b) from happening.
#if __GNUC__ == 4 && (__GNUC_MINOR__ == 1 || __GNUC_MINOR__ == 2)
#define ASSIGN_EVENT(event,type,args) do {type tmp_event args; event = tmp_event;} while(0)
#else
#define ASSIGN_EVENT(event,type,args) event = type args
#endif // __GNUC__ == 4 && (__GNUC_MINOR__ == 1 || __GNUC_MINOR__ == 2)
#endif // __GNUC__

// xlC seems to need this hack also
#ifdef __IBMCPP__
#define ASSIGN_EVENT(event,type,args) do {type tmp_event args; event = tmp_event;} while(0)
#endif

#ifdef BROKEN_COMPARE_AND_SWAP

bool __sync_bool_compare_and_swap( int *ptr, int oldval, int newval );

#endif

// For old machines that do not define CPU_SET macros
#if !__GLIBC_PREREQ (2, 7)
# define __CPU_OP_S(setsize, destset, srcset1, srcset2, op) \
   (__extension__                                                              \
    ({ cpu_set_t *__dest = (destset);                                          \
     __const __cpu_mask *__arr1 = (srcset1)->__bits;                         \
     __const __cpu_mask *__arr2 = (srcset2)->__bits;                         \
     size_t __imax = (setsize) / sizeof (__cpu_mask);                        \
     size_t __i;                                                             \
     for (__i = 0; __i < __imax; ++__i)                                      \
     ((__cpu_mask *) __dest->__bits)[__i] = __arr1[__i] op __arr2[__i];    \
     __dest; }))

# define __CPU_EQUAL_S(setsize, cpusetp1, cpusetp2) \
   (__extension__                                                              \
    ({ __const __cpu_mask *__arr1 = (cpusetp1)->__bits;                        \
     __const __cpu_mask *__arr2 = (cpusetp2)->__bits;                        \
     size_t __imax = (setsize) / sizeof (__cpu_mask);                        \
     size_t __i;                                                             \
     for (__i = 0; __i < __imax; ++__i)                                      \
     if (__arr1[__i] != __arr2[__i])                                       \
     break;                                                              \
     __i == __imax; }))

# define CPU_AND(destset, srcset1, srcset2) \
  __CPU_OP_S (sizeof (cpu_set_t), destset, srcset1, srcset2, &)

# define CPU_OR(destset, srcset1, srcset2) \
  __CPU_OP_S (sizeof (cpu_set_t), destset, srcset1, srcset2, |)

# define CPU_EQUAL(cpusetp1, cpusetp2) \
  __CPU_EQUAL_S (sizeof (cpu_set_t), cpusetp1, cpusetp2)
#endif /* GLIBC < 2.7 */

#if !__GLIBC_PREREQ (2, 6)
#include <limits.h>
inline int __sched_cpucount (size_t setsize, const cpu_set_t *setp)
{
   int s = 0;
   const __cpu_mask *p = setp->__bits;
   const __cpu_mask *end = &setp->__bits[setsize / sizeof (__cpu_mask)];

   while (p < end)
   {
      __cpu_mask l = *p++;

      if (l == 0)
         continue;

# if LONG_BIT > 32
      l = (l & 0x5555555555555555ul) + ((l >> 1) & 0x5555555555555555ul);
      l = (l & 0x3333333333333333ul) + ((l >> 2) & 0x3333333333333333ul);
      l = (l & 0x0f0f0f0f0f0f0f0ful) + ((l >> 4) & 0x0f0f0f0f0f0f0f0ful);
      l = (l & 0x00ff00ff00ff00fful) + ((l >> 8) & 0x00ff00ff00ff00fful);
      l = (l & 0x0000ffff0000fffful) + ((l >> 16) & 0x0000ffff0000fffful);
      l = (l & 0x00000000fffffffful) + ((l >> 32) & 0x00000000fffffffful);
# else
      l = (l & 0x55555555ul) + ((l >> 1) & 0x55555555ul);
      l = (l & 0x33333333ul) + ((l >> 2) & 0x33333333ul);
      l = (l & 0x0f0f0f0ful) + ((l >> 4) & 0x0f0f0f0ful);
      l = (l & 0x00ff00fful) + ((l >> 8) & 0x00ff00fful);
      l = (l & 0x0000fffful) + ((l >> 16) & 0x0000fffful);
# endif

      s += l;
   }

   return s;
}

# define __CPU_COUNT_S(setsize, cpusetp) \
  __sched_cpucount (setsize, cpusetp)

# define CPU_COUNT(cpusetp)      __CPU_COUNT_S (sizeof (cpu_set_t), cpusetp)

inline int sched_getcpu (void)
{
#ifdef __NR_getcpu
   unsigned int cpu;
   int r = getcpu( &cpu, NULL, NULL );
   return r == -1 ? r : cpu;
#else
   return -1;
#endif
}
#endif /* GLIBC < 2.6 */

#endif

