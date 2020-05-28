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
#   AX_CHECK_PAPI
#
# DESCRIPTION
#
#   Check hardware counter support with PAPI
#

AC_DEFUN([AX_CHECK_PAPI],
[
   PAPI_INC=""
   PAPI_LIB=""

   AC_MSG_CHECKING([for PAPI])
   AC_ARG_WITH([papi],
      [AS_HELP_STRING([--with-papi@<:@=DIR@:>@], [build with PAPI library support])],
      [], dnl Implicit: with_papi=$withvalue
      [with_papi=no]
   )
   AC_MSG_RESULT([$with_papi])

   AS_IF([test "x$with_papi" != xno], [
      AS_IF([test -d "$with_papi"], [
         AS_IF([test -d "$with_papi/include"], [papicppflags="-I$with_papi/include"])
         AS_IF([test -d "$with_papi/lib"], [papildflags="-L$with_papi/lib"])
      ])

      ### PAPI INCLUDES
      AC_LANG_PUSH([C++])
      AX_VAR_PUSHVALUE([CPPFLAGS], [$papicppflags])
      AC_CHECK_HEADERS([papi.h], [
         # header found, do nothing
      ], [
         AS_IF([test "x$with_papi" != xcheck], [AC_MSG_ERROR([Cannot find PAPI headers])])
         with_papi=no
      ])
      AX_VAR_POPVALUE([CPPFLAGS])
      AC_LANG_POP([C++])
   ])

   AS_IF([test "x$with_papi" != xno], [
      ### PAPI LIBS
      AC_LANG_PUSH([C++])
      AX_VAR_PUSHVALUE([LIBS], [""])
      AX_VAR_PUSHVALUE([LDFLAGS], [$papildflags])
      AC_SEARCH_LIBS([PAPI_library_init], [papi], [
         papilibs="$LIBS"
      ], [
         AS_IF([test "x$with_papi" != xcheck], [AC_MSG_ERROR([Cannot find PAPI libraries])])
         with_papi=no
      ])
      AX_VAR_POPVALUE([LDFLAGS])
      AX_VAR_POPVALUE([LIBS])
      AC_LANG_POP([C++])
   ])

   AS_IF([test "x$with_papi" != xno], [
      AC_DEFINE([PAPI],[],[Enables PAPI support.])
      AC_SUBST([HAVE_PAPI], [PAPI])
      PAPI_INC="$withval/include"
      PAPI_LIB="$withval/lib"
   ], [
      AC_SUBST([HAVE_PAPI], [NO_PAPI])
   ])
   
   AC_SUBST([PAPI_INC])
   AC_SUBST([PAPI_LIB])
   
   # If papi is present, compile the tgdump instrumentation plugin
   AM_CONDITIONAL([instrumentation_TGDUMP], test "x$with_papi" != xno)
])
