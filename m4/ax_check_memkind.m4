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
#   AX_CHECK_MEMKIND
#
# DESCRIPTION
#
#   Check whether Memkind path to the headers and libraries are correctly specified.
#   Also checks Jemalloc library availability (required by libmemkind).
#

AC_DEFUN([AX_CHECK_MEMKIND],[

AC_ARG_WITH(jemalloc,
[AS_HELP_STRING([--with-jemalloc,--with-jemalloc=PATH],
                [search in system directories or specify prefix directory for installed jemalloc package.])],
[],
[with_jemalloc=no])

AC_ARG_WITH(memkind,
[AS_HELP_STRING([--with-memkind,--with-memkind=PATH],
                [search in system directories or specify prefix directory for installed memkind package.])],
[],
[with_memkind=no])

AS_IF([test "$with_memkind" != no],[

  memkindinc=-I$with_memkind/include
  memkindlib="-L$with_jemalloc/lib -Wl,-rpath,$with_jemalloc/lib -L$with_memkind/lib -Wl,-rpath,$with_memkind/lib"
  
  AC_LANG_PUSH([C++])

  AX_VAR_PUSHVALUE([CPPFLAGS],[$CPPFLAGS $memkindinc])
  AX_VAR_PUSHVALUE([CXXFLAGS])
  AX_VAR_PUSHVALUE([LDFLAGS],[$LDFLAGS $memkindlib])
  AX_VAR_PUSHVALUE([LIBS],[])

  AC_SEARCH_LIBS([je_malloc], [jemalloc], [
    jemalloc=yes
  ],[
    AC_SEARCH_LIBS([jemk_malloc], [jemalloc],
      [jemalloc=yes],
      [jemalloc=no])
  ])

  AS_IF([test "$jemalloc" = yes],[

    AC_CHECK_HEADERS([memkind.h],
      [memkind=yes],
      [memkind=no])

    AS_IF([test "$memkind" = yes],[
      AC_SEARCH_LIBS([memkind_malloc], [memkind],
        [memkind=yes],
        [memkind=no])
    ])dnl

  ],[
    AC_MSG_ERROR([
------------------------------
Could not find libjemalloc (required by memkind)
Please, check that the provided directories are correct.
------------------------------])
  ])dnl

  AS_IF([test "$memkind" = yes],[
    AC_DEFINE([MEMKIND_SUPPORT],[],[Enables memkind support])
  ],[
    AC_MSG_ERROR([
------------------------------
Memkind path was not correctly specified. 
Please, check that the provided directories are correct.
------------------------------])
  ])dnl

  memkindlibs=$LIBS

  AX_VAR_POPVALUE([CPPFLAGS])
  AX_VAR_POPVALUE([CXXFLAGS])
  AX_VAR_POPVALUE([LDFLAGS])
  AX_VAR_POPVALUE([LIBS])

  AC_LANG_POP([C++])

]) dnl with_memkind

AM_CONDITIONAL([MEMKIND_SUPPORT], [test "$memkind" = yes])
AC_SUBST([memkindinc])
AC_SUBST([memkindlib])
AC_SUBST([memkindlibs])

])dnl AX_CHECK_MEMKIND
