# ===========================================================================
#   http://www.gnu.org/software/autoconf-archive/ax_check_compile_flag.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_CHECK_CUDA(FLAG, [ACTION-SUCCESS], [ACTION-FAILURE], [EXTRA-FLAGS], [INPUT])
#
# DESCRIPTION
#
#   Check whether CUDA path to the headers and libraries are correctly specified.
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

AC_DEFUN([AX_CHECK_CUDA],
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

#Check if an CUDA implementation is installed.
AC_ARG_WITH(cuda,
[AS_HELP_STRING([--with-cuda=PATH],
                [specify prefix directory for installed cuda package)])
AC_ARG_WITH(cuda-include,
[AS_HELP_STRING([--with-cuda-include=PATH],
                [specify directory for installed cuda include files])])
AC_ARG_WITH(cuda-lib,
[AS_HELP_STRING([--with-cuda-lib=PATH],
                [specify directory for the installed cuda library])])

if test "x$with_cuda" != xno; then
  if test "x$with_cuda" = xyes; then
    cuda_prefix=/usr/local/cuda
  else
    cuda_prefix=$with_cuda
  fi
  cudainc="-I$with_cuda/include"
  AC_CHECK_FILE([$with_cuda/lib64],
    [cudalib=$with_cuda/lib64],
    [cudalib=$with_cuda/lib])
fi
if test "x$with_cuda_include" != x; then
  cudainc="-I$with_cuda_include"
fi
if test "x$with_cuda_lib" != x; then
  cudalib="-L$with_cuda_lib"
fi

# This is fulfilled even if $with_cuda="yes" 
# This happens when user leaves --with-value empty
if test x$with_cuda$with_cuda_include$with_cuda_lib != x; then

    # Check for Nvidia Cuda Compiler NVCC
    AC_PATH_PROG([NVCC], [nvcc], [], [$cuda_prefix/bin$PATH_SEPARATOR$PATH])

    #tests if provided headers and libraries are usable and correct
    bak_CFLAGS="$CFLAGS"
    bak_CPPFLAGS="$CPPFLAGS"
    bak_LDFLAGS="$LDFLAGS"

    CFLAGS=
    CPPFLAGS=$cudainc
    LDFLAGS=$cudalib

    # One of the following two header files has to exist
    AC_CHECK_HEADER([cuda.h], [cuda=yes])
    # Look for clGetPlatformIDs function in either libmali.so or libcuda.so libraries
    AC_SEARCH_LIBS([cudaMemcpy],
                   [cudart],
                   [cuda=yes])

    CFLAGS="$bak_CFLAGS"
    CPPFLAGS="$bak_CPPFLAGS"
    LDFLAGS="$bak_LDFLAGS"

    if test x$cuda == xno; then
        AC_MSG_ERROR([
------------------------------
CUDA path was not correctly specified. 
Please, check that the provided directories are correct.
------------------------------])
    fi

    AC_MSG_CHECKING([CUDA API version])
    cuda_version = $(grep "define CUDA_VERSION $cudainc/cuda.h" | sed 's-/\*.*/--g' | sed 's/[^0-9]//g')
    AC_MSG_RESULT([$cuda_version])
    if test $cuda_version -lt 5000; then
        AC_MSG_ERROR([
------------------------------
Version of the provided CUDA package is too old.
CUDA 5 or greater is required.
------------------------------])
    fi
fi

if test x$cuda = xyes; then
    ARCHITECTURES="$ARCHITECTURES gpu"

    cuda_rpath="-Xlinker -rpath -Xlinker $CUDA_LIB"

    AC_DEFINE_UNQUOTED([NANOS_CUDA_VERSION],[$cuda_version],[API version of the CUDA package specified by the user])
    AC_DEFINE([GPU_DEV])
    CPPFLAGS="$CPPFLAGS -isystem $CUDA_INC"
    LDFLAGS="$LDFLAGS $cuda_rpath -L$cudalib -lcudart"
    
    #CFLAGS="$CFLAGS -DGPU_DEV -DNANOS_CUDA_VERSION=$cuda_version -isystem $CUDA_INC"
    #CXXFLAGS="$CXXFLAGS -DGPU_DEV -DNANOS_CUDA_VERSION=$cuda_version -isystem $CUDA_INC"
    #LDFLAGS="$LDFLAGS $CUDA_RPATH -L$CUDA_LIB -lcudart"

    AC_SUBST([cudainc])
    AC_SUBST([cudalib])
    AC_SUBST([cuda_rpath])
fi

AC_SUBST([cuda])
AC_SUBST([cuda_prefix])
AM_CONDITIONAL([GPU_SUPPORT],[test x$cuda = xyes])

])dnl AX_CHECK_COMPILE_FLAGS
