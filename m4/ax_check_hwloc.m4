# ===========================================================================
#   http://www.gnu.org/software/autoconf-archive/ax_check_compile_flag.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_CHECK_HWLOC
#
# DESCRIPTION
#
#   Check whether hwloc path to the headers and libraries are correctly specified.
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

AC_DEFUN([AX_CHECK_HWLOC],[
  AC_PREREQ(2.59)dnl for _AC_LANG_PREFIX
  
  #Check if an HWLOC implementation is installed.
  AC_ARG_WITH(hwloc,
  [AS_HELP_STRING([--with-hwloc=PATH],
                  [specify prefix directory for installed hwloc package])])
  AC_ARG_WITH(hwloc-include,
  [AS_HELP_STRING([--with-hwloc-include=PATH],
                  [specify directory for installed hwloc include files])])
  AC_ARG_WITH(hwloc-lib,
  [AS_HELP_STRING([--with-hwloc-lib=PATH],
                  [specify directory for the installed hwloc library])])
  
  if test "x$with_hwloc" != xno; then
    if test "x$with_hwloc" = xyes; then
      hwloc_prefix=/usr/local/hwloc
    else
      hwloc_prefix=$with_hwloc
    fi
    hwlocinc="-I $with_hwloc/include"
    hwloc_h="$with_hwloc/include/hwloc.h"
    AC_CHECK_FILE([$with_hwloc/lib64],
      [hwloclib=-L$with_hwloc/lib64],
      [hwloclib=-L$with_hwloc/lib])
  fi
  if test "x$with_hwloc_include" != x; then
    hwlocinc="-isystem $with_hwloc_include"
    hwloc_h="$with_hwloc_include/hwloc.h"
  fi
  if test "x$with_hwloc_lib" != x; then
    hwloclib="-L$with_hwloc_lib"
  fi
  
  # This is fulfilled even if $with_hwloc="yes" 
  # This happens when user leaves --with-value alone
  if test x$with_hwloc$with_hwloc_include$with_hwloc_lib != x; then
  
      #tests if provided headers and libraries are usable and correct
      bak_CFLAGS="$CFLAGS"
      bak_CPPFLAGS="$CPPFLAGS"
      bak_LIBS="$LIBS"
      bak_LDFLAGS="$LDFLAGS"
  
      CFLAGS=
      CPPFLAGS=$hwlocinc
      LIBS=
      LDFLAGS=$hwloclib
  
      # Check if hwloc.h header file exists and compiles
      AC_CHECK_HEADER([hwloc.h], [hwloc=yes],[hwloc=no])

      # Look for hwlocMemcpy function in libhwlocrt.so library
      if test x$hwloc == xyes; then
        AC_CHECK_LIB([hwloc],
                       [hwloc_topology_init],
                       [hwloc=yes],[hwloc=no])
      fi

      hwloclib="$hwloclib $LIBS"
  
      CFLAGS="$bak_CFLAGS"
      CPPFLAGS="$bak_CPPFLAGS"
      LIBS="$bak_LIBS"
      LDFLAGS="$bak_LDFLAGS"
  
      if test x$hwloc != xyes; then
          AC_MSG_ERROR([
------------------------------
hwloc path was not correctly specified. 
Please, check that the provided directories are correct.
------------------------------])
      fi
  
      AC_CACHE_CHECK([HWLOC API version],[ac_cv_hwloc_version],
        [
          ac_cv_hwloc_version=$(grep 'define HWLOC_API_VERSION' "$hwloc_h")
          ac_cv_hwloc_version=$(expr "x$ac_cv_hwloc_version" : 'x#define HWLOC_API_VERSION \(0x[0-9a-fA-F]*\)')
        ])

    if test x$ac_cv_hwloc_version != xyes -o $ac_cv_hwloc_version -lt 0x010200; then
      AC_MSG_ERROR([
------------------------------
Version of the provided hwloc package is too old.
hwloc 1.2.0 or greater is required.
------------------------------])
    fi
  fi
  
  if test x$hwloc = xyes; then
      OPTIONS="$OPTIONS hwloc"
  
      AC_DEFINE_UNQUOTED([NANOS_HWLOC_VERSION],[$hwloc_version],[Version of the hwloc package specified by the user])
      AC_DEFINE([HWLOC],[],[Indicates the presence of hwloc library.])
  fi
  
  AC_SUBST([hwloc])
  AC_SUBST([hwlocinc])
  AC_SUBST([hwloclib])

  AM_CONDITIONAL([HWLOC],[test x$hwloc = xyes])
])dnl AX_CHECK_HWLOC
