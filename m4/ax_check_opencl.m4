# ===========================================================================
#   http://www.gnu.org/software/autoconf-archive/ax_check_compile_flag.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_CHECK_OPENCL(FLAG, [ACTION-SUCCESS], [ACTION-FAILURE], [EXTRA-FLAGS], [INPUT])
#
# DESCRIPTION
#
#   Check whether OpenCL path to the headers and libraries are correctly specified.
#
#   ACTION-SUCCESS/ACTION-FAILURE are shell commands to execute on
#   success/failure.
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

AC_DEFUN([AX_CHECK_OPENCL],
[
# AC_PREREQ(2.59)dnl for _AC_LANG_PREFIX
# AS_VAR_PUSHDEF([CACHEVAR],[ax_cv_check_[]_AC_LANG_ABBREV[]flags_$4_$1])dnl
# AC_CACHE_CHECK([whether _AC_LANG compiler accepts $1], CACHEVAR, [
#   ax_check_save_flags=$[]_AC_LANG_PREFIX[]FLAGS
#   _AC_LANG_PREFIX[]FLAGS="$[]_AC_LANG_PREFIX[]FLAGS $4 $1"
#   AC_COMPILE_IFELSE([m4_default([$5],[AC_LANG_PROGRAM()])],
#     [AS_VAR_SET(CACHEVAR,[yes])],
#     [AS_VAR_SET(CACHEVAR,[no])])
#   _AC_LANG_PREFIX[]FLAGS=$ax_check_save_flags])
# AS_IF([test x"AS_VAR_GET(CACHEVAR)" = xyes],
#   [m4_default([$2], :)],
#   [m4_default([$3], :)])
# AS_VAR_POPDEF([CACHEVAR])dnl
AC_PREREQ(2.59)dnl for _AC_LANG_PREFIX

#Check if an OpenCL implementation is installed.
AC_ARG_WITH(opencl,
[AS_HELP_STRING([--with-opencl=PATH],
                [specify prefix directory for installed OpenCL package.
                 Equivalent to --with-opencl-include=PATH/include
                 plus --with-opencl-lib=PATH/lib])])
AC_ARG_WITH(opencl-include,
[AS_HELP_STRING([--with-opencl-include=PATH],
                [specify directory for installed OpenCL include files])])
AC_ARG_WITH(opencl-lib,
[AS_HELP_STRING([--with-opencl-lib=PATH],
                [specify directory for the installed OpenCL library])])

if test "x$with_opencl" != xyes -a "x$with_opencl" != xno; then
  openclinc="-I$with_opencl/include"
  opencllib="-L$with_opencl/lib"
fi
if test "x$with_opencl_include" != x; then
  openclinc="-I$with_opencl_include"
fi
if test "x$with_opencl_lib" != x; then
  opencllib="-L$with_opencl_lib"
fi

# This is fulfilled even if $with_opencl="yes" 
# This happens when user leaves --with-value empty
# In this case, both openclinc and opencllib will be empty
# so the test should search in default locations and LD_LIBRARY_PATH
if test "x$with_opencl" != xno -a "x$with_opencl$with_opencl_include$with_opencl_lib" != x; then
    #tests if provided headers and libraries are usable and correct
    bak_CFLAGS="$CFLAGS"
    bak_CPPFLAGS="$CPPFLAGS"
    bak_LDFLAGS="$LDFLAGS"

    CFLAGS=
    CPPFLAGS=$openclinc
    LDFLAGS=$opencllib

    # One of the following two header files has to exist
    AC_CHECK_HEADER([CL/opencl.h], [opencl=yes])
    AC_CHECK_HEADER([OpenCL/opencl.h], [opencl=yes])
    # Look for clGetPlatformIDs function in either libmali.so or libOpenCL.so libraries
    AC_SEARCH_LIBS([clGetPlatformIDs],
                   [OpenCL mali],
                   [opencl=yes])

    CFLAGS="$bak_CFLAGS"
    CPPFLAGS="$bak_CPPFLAGS"
    LDFLAGS="$bak_LDFLAGS"

    if test x$opencl == xno; then
        AC_MSG_ERROR([OpenCL was not found. Please, check that the provided directories are correct.])
    fi
fi

if test x$opencl = xyes; then
    OPENCL_LD=$OPENCL_LIB
    OPENCL_INC=$OPENCL_INC
    CFLAGS="$CFLAGS -L$OPENCL_LIB -isystem $OPENCL_INC -DOpenCL_DEV"
    CXXFLAGS="$CXXFLAGS -L$OPENCL_LIB -isystem $OPENCL_INC -DOpenCL_DEV"
    ARCHITECTURES="$ARCHITECTURES opencl"
    AC_SUBST([OPENCL_LD])
    AC_SUBST([OPENCL_INC])
fi

AM_CONDITIONAL([OPENCL_SUPPORT],[test x$opencl = xyes])

AC_SUBST([opencl])
])dnl AX_CHECK_COMPILE_FLAGS
