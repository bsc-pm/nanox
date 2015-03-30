#
# SYNOPSIS
#
#   AX_CHECK_CUDA
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

AC_DEFUN([AX_CHECK_CUDA],[
  AC_PREREQ(2.59)dnl for _AC_LANG_PREFIX
  
  #Check if an CUDA implementation is installed.
  AC_ARG_WITH(cuda,
  [AS_HELP_STRING([--with-cuda,--with-cuda=PATH],
                  [search in system directories or specify prefix directory for installed cuda package])])
  AC_ARG_WITH(cuda-include,
  [AS_HELP_STRING([--with-cuda-include=PATH],
                  [specify directory for installed cuda include files])])
  AC_ARG_WITH(cuda-lib,
  [AS_HELP_STRING([--with-cuda-lib=PATH],
                  [specify directory for the installed cuda library])])
  
  # Search for Cuda by default
  if test "x$with_cuda" != xno; then
    if [[[ "x$with_cuda" =~ x"yes"|"" ]]]; then
      cuda_prefix=/usr/local/cuda
    else
      cuda_prefix=$with_cuda
    fi
    cudainc="-isystem $cuda_prefix/include"
    cuda_h="$cuda_prefix/include/cuda.h"
    AC_CHECK_FILE([$cuda_prefix/lib64],
      [cudalib=-L$cuda_prefix/lib64 -Wl,-rpath=$cuda_prefix/lib64],
      [cudalib=-L$cuda_prefix/lib -Wl,rpath=$cuda_prefix/lib])
  fi
  if test "x$with_cuda_include" != x; then
    cudainc="-isystem $with_cuda_include"
    cuda_h="$with_cuda_include/cuda.h"
  fi
  if test "x$with_cuda_lib" != x; then
    cudalib="-L$with_cuda_lib"
  fi
  
  # This is fulfilled even if $with_cuda="yes" 
  # This happens when user leaves --with-value alone
  if test x$with_cuda$with_cuda_include$with_cuda_lib != x; then
  
      # Check for Nvidia Cuda Compiler NVCC
      AC_PATH_PROG([NVCC], [nvcc], [], [$cuda_prefix/bin$PATH_SEPARATOR$PATH])
  
      #tests if provided headers and libraries are usable and correct
      bak_CFLAGS="$CFLAGS"
      bak_CPPFLAGS="$CPPFLAGS"
      bak_LIBS="$LIBS"
      bak_LDFLAGS="$LDFLAGS"
  
      CFLAGS=
      CPPFLAGS=$cudainc
      LIBS=
      LDFLAGS=$cudalib
  
      # Check if cuda.h header file exists and compiles
      AC_CHECK_HEADER([cuda.h], [cuda=yes],[cuda=no])

      # Look for cudaMemcpy function in libcudart.so library
      if test x$cuda == xyes; then
        AC_CHECK_LIB([cudart],
                       [cudaMemcpy],
                       [cuda=yes
                        LIBS="$LIBS -lcudart"],
                       [cuda=no])
      fi

      # Look for cublasDrotmg function in libcublas.so library
      if test x$cuda == xyes; then
        AC_CHECK_LIB([cublas],
                       [cublasDrotmg],
                       [cuda=yes
                        LIBS="$LIBS -lcublas"],
                       [cuda=no],
                       [-lcudart])
      fi

      cudalibs=$LIBS
  
      CFLAGS="$bak_CFLAGS"
      CPPFLAGS="$bak_CPPFLAGS"
      LIBS="$bak_LIBS"
      LDFLAGS="$bak_LDFLAGS"
  
      if test x$cuda != xyes; then
          AC_MSG_ERROR([
------------------------------
CUDA path was not correctly specified. 
Please, check that the provided directories are correct.
------------------------------])
      fi
  
      AC_CACHE_CHECK([CUDA API version],[ac_cv_cuda_version],
        [
          ac_cv_cuda_version=$(grep 'define CUDA_VERSION' "$cuda_h")
          ac_cv_cuda_version=$(expr "x$ac_cv_cuda_version" : 'x#define CUDA_VERSION \(@<:@0-9@:>@*\)')
        ])

      if test "x$ac_cv_cuda_version" == "x" -o "$ac_cv_cuda_version" -lt 5000; then
          AC_MSG_ERROR([
------------------------------
Version of the provided CUDA package is too old.
CUDA 5 or greater is required.
------------------------------])
      fi
  fi
  
  if test x$cuda = xyes; then
      ARCHITECTURES="$ARCHITECTURES gpu"
  
      AC_DEFINE_UNQUOTED([NANOS_CUDA_VERSION],[$cuda_version],[API version of the CUDA package specified by the user])
      AC_DEFINE([GPU_DEV],[],[Indicates the presence of the GPU arch plugin.])
  fi
  
  AC_SUBST([cuda])
  AC_SUBST([cuda_prefix])
  AC_SUBST([cudainc])
  AC_SUBST([cudalib])
  AC_SUBST([cudalibs])

  AM_CONDITIONAL([GPU_SUPPORT],[test x$cuda = xyes])
])dnl AX_CHECK_CUDA
