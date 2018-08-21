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
#   AX_CHECK_HWLOC
#
# DESCRIPTION
#
#   Check whether hwloc path to the headers and libraries are correctly specified.
#

AC_DEFUN([AX_CHECK_HWLOC],[
AC_PREREQ(2.59)dnl for _AC_LANG_PREFIX

#Check if an HWLOC implementation is installed.
AC_ARG_WITH(hwloc,
[AS_HELP_STRING([--with-hwloc,--with-hwloc=PATH],
                [search in system directories or specify prefix directory for installed hwloc package])])
AC_ARG_WITH(hwloc-include,
[AS_HELP_STRING([--with-hwloc-include=PATH],
                [specify directory for installed hwloc include files])])
AC_ARG_WITH(hwloc-lib,
[AS_HELP_STRING([--with-hwloc-lib=PATH],
                [specify directory for the installed hwloc library])])

AS_IF([test "x$with_hwloc" != xyes dnl
         -o "x$with_hwloc" != xno dnl
         -o "x$with_hwloc" != x dnl
],[
  AS_IF([test -d $with_hwloc/include], [
    hwlocinc="-I $with_hwloc/include"
    hwloc_h="$with_hwloc/include/hwloc.h"
  ])
  AS_IF([test -d $with_hwloc/lib64], [
    hwloclib="-L$with_hwloc/lib64 -Wl,-rpath,$with_hwloc/lib64"
  ], [test -d $with_hwloc/lib], [
    hwloclib="-L$with_hwloc/lib -Wl,-rpath,$with_hwloc/lib"
  ])dnl
])dnl

AS_IF([test "x$with_hwloc_lib" != x],[
  hwloclib="-L$with_hwloc_lib -Wl,-rpath,$with_hwloc_lib"
])

# This condition is satisfied even if $with_hwloc="yes" 
# This happens when user leaves --with-value alone
AS_IF([test "x$with_hwloc$with_hwloc_lib" != x],[
  AC_LANG_PUSH([C++])

  #tests if provided headers and libraries are usable and correct
  AX_VAR_PUSHVALUE([CPPFLAGS],[$CPPFLAGS $hwlocinc])
  AX_VAR_PUSHVALUE([CXXFLAGS])
  AX_VAR_PUSHVALUE([LDFLAGS],[$LDFLAGS $hwloclib])
  AX_VAR_PUSHVALUE([LIBS],[])

  # Check if hwloc.h header file exists and compiles
  AC_CHECK_HEADERS([hwloc.h], [hwloc=yes],[hwloc=no])

  # Look for hwloc_get_api_version function in libhwloc.so library
  AS_IF([test "x$hwloc" = xyes],[
    AC_CHECK_LIB([hwloc],
             [hwloc_get_api_version],
             [hwloc=yes
              LIBS=-lhwloc],
             [hwloc=no])
  ])dnl

  AS_IF([test x$hwloc != xyes],[
      AC_MSG_ERROR([
------------------------------
hwloc path was not correctly specified. 
Please, check that the provided directories are correct.
------------------------------])
  ])dnl

  AS_IF([test x$hwloc = xyes],[
    AC_CACHE_CHECK([hwloc version],[ac_cv_hwloc_version],
      [AC_RUN_IFELSE(
        [AC_LANG_PROGRAM(
          [
             #ifdef HAVE_HWLOC_H
                 #include <hwloc.h>
             #endif
             #include <stdio.h>
             #include <stdlib.h>
          ],
          [
             unsigned hwloc_version = hwloc_get_api_version();
 				
             FILE* out = fopen("conftest.out","w");
             fprintf(out,"hwloc 0x%x\n", hwloc_version);
             return fclose(out);
          ])],
        [ac_cv_hwloc_version=$(cat conftest.out)
        ],
        [AC_MSG_FAILURE([
------------------------------
hwloc version test execution failed
------------------------------])
        ],
        [
          ac_cv_hwloc_version=skip
        ])
    ])

    AS_IF([test "$ac_cv_hwloc_version" != skip],[
      ac_cv_hwloc_version=$(expr "x$ac_cv_hwloc_version" : 'xhwloc \(0x@<:@0-9a-f@:>@*\)$')
    ])
  ])dnl hwloc 

  AS_IF([test "$ac_cv_hwloc_version" = skip],[
    AC_MSG_WARN([
------------------------------
Hwloc library version cannot be checked
because cross-compilation mode has been detected.
------------------------------])
  ], [
    AS_IF([test "x$ac_cv_hwloc_version" != x],[
      AS_IF([(("$ac_cv_hwloc_version" < 0x010200))],[
        AC_MSG_ERROR([
------------------------------
Version of the provided hwloc package is too old ($ac_cv_hwloc_version).
hwloc 1.2.0 or greater is required.
------------------------------])
      ])dnl
  ],[
      AC_MSG_ERROR([
------------------------------
Could not find hwloc package version @{:@value: $ac_cv_hwloc_version@:}@.
Check config.log for details.
------------------------------])
    ])dnl
  ])dnl

  hwloclibs=$LIBS

  AX_VAR_POPVALUE([CPPFLAGS])
  AX_VAR_POPVALUE([CXXFLAGS])
  AX_VAR_POPVALUE([LDFLAGS])
  AX_VAR_POPVALUE([LIBS])

  AC_LANG_POP([C++])

])dnl with_hwloc

AS_IF([test "$hwloc" = yes],[
  OPTIONS="$OPTIONS hwloc"
  AC_DEFINE_UNQUOTED([NANOS_HWLOC_VERSION],[$hwloc_version],[Version of the hwloc package specified by the user])
  AC_DEFINE([HWLOC],[],[Indicates the presence of hwloc library.])
  AC_SUBST([HAVE_HWLOC], [HWLOC])
],[
  AC_SUBST([HAVE_HWLOC], [NO_HWLOC])
  hwlocinc=
  hwloclib=
  hwloclibs=
])dnl

AC_SUBST([hwloc])
AC_SUBST([hwlocinc])
AC_SUBST([hwloclib])
AC_SUBST([hwloclibs])

AM_CONDITIONAL([HWLOC],[test x$hwloc = xyes])
])dnl AX_CHECK_HWLOC
