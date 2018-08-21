#####################################################################################
#      Copyright 2009-2018 Barcelona Supercomputing Center                          #
#                                                                                   #
#      This file is part of the NANOS++ library.                                    #
#                                                                                   #
#      NANOS++ is free software: you can redistribute it and/or modify              #
#      it under the terms of the GNU Lesser General Public License as published by  #
#      the Free Software Foundation, either version 3 of the License, or            #
#      (at your option) any later version.                                          #
#                                                                                   #
#      NANOS++ is distributed in the hope that it will be useful,                   #
#      but WITHOUT ANY WARRANTY; without even the implied warranty of               #
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                #
#      GNU Lesser General Public License for more details.                          #
#                                                                                   #
#      You should have received a copy of the GNU Lesser General Public License     #
#      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            #
#####################################################################################

#
# SYNOPSIS
#
#   AC_CHECK_GXX_NEW_ATOMIC_BUILTINS
#
# DESCRIPTION
#
#   Check whether the new GCC atomic builtins are enabled
#

AC_DEFUN([AC_CHECK_GXX_NEW_ATOMIC_BUILTINS], [
    new_gcc_builtins=no
    try_gcc_builtins=no
    AC_ARG_ENABLE([gcc-new-atomic-builtins],
      AS_HELP_STRING([--enable-gcc-new-atomic-builtins], [Tries to use gcc (>=4.7) new atomic builtins]),
      [
      try_gcc_builtins=$enableval
      ],
      [])
    AC_MSG_CHECKING([for new atomic builtins in GCC])
    if test "$try_gcc_builtins" != no;
    then
      AC_LANG_PUSH([C++])
      AC_LINK_IFELSE(
        [AC_LANG_PROGRAM([],
          [[
          int a = 1, b = 1;
          __atomic_fetch_add(&a, 1, __ATOMIC_ACQ_REL);
          __atomic_fetch_sub(&a, 1, __ATOMIC_ACQ_REL);
          __atomic_add_fetch(&a, 1, __ATOMIC_ACQ_REL);
          __atomic_sub_fetch(&a, 1, __ATOMIC_ACQ_REL);
          __atomic_compare_exchange_n(&a, &b, 2, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
          __atomic_exchange_n(&a, 1, __ATOMIC_ACQ_REL);
          __atomic_load_n(&a, __ATOMIC_ACQUIRE);
          __atomic_store_n(&a, 1, __ATOMIC_RELEASE);
          __atomic_thread_fence(__ATOMIC_ACQ_REL);
          ]]
          )],
        [
            new_gcc_builtins=yes
            AC_DEFINE([HAVE_NEW_GCC_ATOMIC_OPS], [1], [Defined if the compiler supports the new gcc atomic builtins])
            AC_MSG_RESULT([yes])
        ],
        [
            new_gcc_builtins=no
            AC_MSG_RESULT([no])
        ]
    )
      AC_LANG_POP([C++])
    else
       AC_MSG_RESULT([disabled])
    fi

    AS_IF([test "$new_gcc_builtins" = yes], [
       AC_SUBST([HAVE_NEW_GCC_ATOMIC_OPS], [HAVE_NEW_GCC_ATOMIC_OPS])
    ], [
       AC_SUBST([HAVE_NEW_GCC_ATOMIC_OPS], [NO_HAVE_NEW_GCC_ATOMIC_OPS])
    ])
])

AC_DEFUN([AC_CHECK_GXX_LEGACY_ATOMIC_BUILTINS], [
   AC_MSG_CHECKING([for legacy atomic builtins in GCC])
   AC_LANG_PUSH([C++])
   AC_LINK_IFELSE(
      [AC_LANG_PROGRAM([],
         [[
          int foo = -10;
          int bar = 10;
          __sync_fetch_and_add(&foo, bar);
         ]]
      )],
      [ac_cv_have_builtins=yes],
      [ac_cv_have_builtins=no])
      if test x"$ac_cv_have_builtins" = xno; then
         CXXFLAGS+=" -march=i686"
         AC_LINK_IFELSE(
            [AC_LANG_PROGRAM([],
               [[
                int foo = -10;
                int bar = 10;
                __sync_fetch_and_add(&foo, bar);
               ]]
            )],
            [ac_cv_have_builtins_insist=yes],
            [ac_cv_have_builtins_insist=no])
         if test x"$ac_cv_have_builtins_insist" = xno; then
            AC_MSG_RESULT([no])
            AC_MSG_ERROR([Gcc atomic builtins are necessary to compile this library])
         else
            AC_MSG_RESULT([yes])
         fi
      else
         AC_MSG_RESULT([yes])
      fi
# __sync_bool_compare_and_swap_8
   AC_MSG_CHECKING([if __sync_bool_compare_and_swap_8 is available])
   AC_LINK_IFELSE(
            [AC_LANG_PROGRAM([],
               [[
                    typedef unsigned long long uint64;
                    uint64 i;
                    __sync_bool_compare_and_swap (&i, 0, 1);
               ]]
            )],
            [available_builtin=yes],
            [available_builtin=no])
   if test "$available_builtin" = "yes"; then
       AC_MSG_RESULT([yes])
       AC_DEFINE(HAVE_SYNC_BOOL_COMPARE_AND_SWAP_8, 1,
       [Define to 1 if the compiler provides the __sync_bool_compare_and_swap function for uint64])
   else
       AC_MSG_RESULT([no])
   fi

# __sync_add_and_fetch_8
   AC_MSG_CHECKING([if __sync_add_and_fetch_8 is available])
   AC_LINK_IFELSE(
      [AC_LANG_PROGRAM([],
               [[
                  typedef unsigned long long uint64;
                  uint64 i;
                  __sync_add_and_fetch (&i, 1);
               ]]
            )],
      [available_builtins=yes], [available_builtins=no])
   if test x$available_builtins = xyes; then
      AC_MSG_RESULT([yes])
      AC_DEFINE(HAVE_SYNC_ADD_AND_FETCH_8, 1, [Define to 1 if the compiler provides the __sync_add_and_fetch function for uint64])
   else
      AC_MSG_RESULT([no])
   fi
# __sync_sub_and_fetch_8
    AC_MSG_CHECKING([if __sync_sub_and_fetch_8 is available])
    AC_LINK_IFELSE(
       [AC_LANG_PROGRAM([],
          [[
             typedef unsigned long long uint64;
             uint64 i;
             __sync_sub_and_fetch (&i, 1);
          ]]
          )],
             [available_builtins=yes],
             [available_builtins=no])
    if test "$available_builtins" = "yes"; then
        AC_MSG_RESULT([yes])
        AC_DEFINE(HAVE_SYNC_SUB_AND_FETCH_8, 1,
        [Define to 1 if the compiler provides the __sync_sub_and_fetch function for uint64])
    else
        AC_MSG_RESULT([no])
    fi
# __sync_or_and_fetch_8
    AC_MSG_CHECKING([if __sync_or_and_fetch_8 is available])
    AC_LINK_IFELSE(
       [AC_LANG_PROGRAM([],
          [[
             typedef unsigned long long uint64;
             uint64 i;
             __sync_or_and_fetch (&i, 1);
          ]]
          )],
             [available_builtins=yes],
             [available_builtins=no])
    if test "$available_builtins" = "yes"; then
        AC_MSG_RESULT([yes])
        AC_DEFINE(HAVE_SYNC_OR_AND_FETCH_8, 1,
        [Define to 1 if the compiler provides the __sync_or_and_fetch function for uint64])
    else
        AC_MSG_RESULT([no])
    fi
# __sync_and_and_fetch_8
    AC_MSG_CHECKING([if __sync_and_and_fetch_8 is available])
    AC_LINK_IFELSE(
       [AC_LANG_PROGRAM([],
          [[
             typedef unsigned long long uint64;
             uint64 i;
             __sync_and_and_fetch (&i, 1);
          ]]
          )],
             [available_builtins=yes],
             [available_builtins=no])
    if test "$available_builtins" = "yes"; then
        AC_MSG_RESULT([yes])
        AC_DEFINE(HAVE_SYNC_AND_AND_FETCH_8, 1,
        [Define to 1 if the compiler provides the __sync_and_and_fetch function for uint64])
    else
        AC_MSG_RESULT([no])
    fi
# __sync_xor_and_fetch_8
    AC_MSG_CHECKING([if __sync_xor_and_fetch_8 is available])
    AC_LINK_IFELSE(
       [AC_LANG_PROGRAM([],
          [[
             typedef unsigned long long uint64;
             uint64 i;
             __sync_xor_and_fetch (&i, 1);
          ]]
          )],
             [available_builtins=yes],
             [available_builtins=no])
    if test "$available_builtins" = "yes"; then
        AC_MSG_RESULT([yes])
        AC_DEFINE(HAVE_SYNC_XOR_AND_FETCH_8, 1,
        [Define to 1 if the compiler provides the __sync_xor_and_fetch function for uint64])
    else
        AC_MSG_RESULT([no])
    fi
   AC_LANG_POP([C++])
])
