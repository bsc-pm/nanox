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
# LICENSE
#
#   Copyright (c) 2015 Jorge Bellon <jbellon@bsc.es>
#
#   This program is free software: you can redistribute it and/or modify it
#   under the terms of the GNU General Public License as published by the
#   Free Software Foundation, either version 3 of the License, or (at your
#   option) any later version.
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
#   Public License for more details.
#
#   You should have received a copy of the GNU General Public License along
#   with this program. If not, see <http://www.gnu.org/licenses/>.
#
#   As a special exception, the respective Autoconf Macro's copyright owner
#   gives unlimited permission to copy, distribute and modify the configure
#   scripts that are the output of Autoconf when processing the Macro. You
#   need not follow the terms of the GNU General Public License when using
#   or distributing such scripts, even though portions of the text of the
#   Macro appear in them. The GNU General Public License (GPL) does govern
#   all other use of the material that constitutes the Autoconf Macro.
#
#   This special exception to the GPL applies to versions of the Autoconf
#   Macro released by the Autoconf Archive. When you make and distribute a
#   modified version of the Autoconf Macro, you may extend this special
#   exception to the GPL to apply to your modified version as well.

AC_DEFUN([AX_CHECK_MEMKIND],[

LDFLAGS_BKP="$LDFLAGS"
LIBS_BKP="$LIBS"

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

if test x$with_memkind != xno; then

  AC_LANG_PUSH([C++])

  save_CPPFLAGS=$CPPFLAGS
  save_CXXFLAGS=$CXXFLAGS
  save_LDFLAGS=$LDFLAGS
  save_LIBS=$LIBS

  LIBS=

  jemalloclib="-L$with_jemalloc/lib -Wl,-rpath,$with_jemalloc/lib"
  
  LDFLAGS="$LDFLAGS $jemalloclib"
  AC_SEARCH_LIBS([je_malloc], [jemalloc],
    [jemalloc=yes], 
    [jemalloc=no])

  if test x$jemalloc = xyes; then
    memkindinc=-I$with_memkind/include
    memkindlib="-L$with_memkind/lib -Wl,-rpath,$with_memkind/lib"

    CPPFLAGS="$CPPFLAGS $memkindinc"
    LDFLAGS="$LDFLAGS $memkindlib"

    AC_CHECK_HEADERS([memkind.h],
      [memkind=yes],
      [memkind=no])

    if test x$memkind = yes; then
      AC_SEARCH_LIBS([memkind_malloc], [memkind],
        [memkind=yes],
        [memkind=no])
    fi

  else
    AC_MSG_ERROR([
------------------------------
Could not find libjemalloc.so (required by memkind)
Please, check that the provided directories are correct.
------------------------------])
  fi

  if test $memkind != yes; then
    AC_MSG_ERROR([
------------------------------
Memkind path was not correctly specified. 
Please, check that the provided directories are correct.
------------------------------])
  fi

fi dnl if memkind

AM_CONDITIONAL([MEMKIND_SUPPORT], [test $memkind = yes])
AC_SUBST([JEMALLOC_LIBS],[jemalloclib])
AC_SUBST([MEMKIND_LIBS],[memkindlibs])
AC_SUBST([MEMKIND_LDFLAGS],[memkindlib])
AC_SUBST([MEMKIND_CFLAGS],[memkindinc])

])
