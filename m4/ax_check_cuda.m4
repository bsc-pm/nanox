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
#   AX_CHECK_CUDA
#
# DESCRIPTION
#
#   Check whether CUDA path to the headers and libraries are correctly specified.
#
#   ACTION-SUCCESS/ACTION-FAILURE are shell commands to execute on
#   success/failure.
#

AC_DEFUN([AX_CHECK_CUDA],[
  AC_PREREQ(2.59)dnl for _AC_LANG_PREFIX

  # Let the user specify CUDA compiler using an environment variable
  AC_ARG_VAR(NVCC,[CUDA compiler command])
  
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

  # Check if user specifically requested this package.
  # It will generate errors if we are not able to find headers/libs
  AS_IF([test "x$with_cuda$with_cuda_include$with_cuda_lib" != x],[
    user_requested="yes"
  ],[
    user_requested="no"
  ])dnl

  # Search for Cuda by default (no --with-cuda specified by the user)
  AS_IF([test "x$with_cuda" != xno],[
    # If user does not specify a PATH, use /usr/local/cuda as default
    AS_IF([test ! -d "$with_cuda"],[
      cuda_prefix=/usr/local/cuda
    ],[
      cuda_prefix=$with_cuda
    ])dnl

    cudainc="-isystem $cuda_prefix/include"
    cuda_h="$cuda_prefix/include/cuda.h"
    AS_IF([test -d $cuda_prefix/lib64],
      [cudalib="-L$cuda_prefix/lib64 -Wl,-rpath,$cuda_prefix/lib64"],
      [cudalib="-L$cuda_prefix/lib -Wl,rpath,$cuda_prefix/lib"])dnl
  ])dnl

  AS_IF([test "x$with_cuda_include" != x],[
    cudainc="-isystem $with_cuda_include"
    cuda_h="$with_cuda_include/cuda.h"
  ])dnl

  AS_IF([test "x$with_cuda_lib" != x],[
    cudalib="-L$with_cuda_lib -Wl,-rpath,$with_cuda_lib"
  ])dnl
  
  # This condition is satisfied even if $with_cuda="yes" 
  # This happens when user leaves --with-value alone
  AS_IF([test x$with_cuda != xno],[
      AC_LANG_PUSH([C++])
  
      # Check for Nvidia Cuda Compiler NVCC
      AC_PATH_PROG([NVCC], [nvcc], [], [$cuda_prefix/bin$PATH_SEPARATOR$PATH])
  
      #tests if provided headers and libraries are usable and correct
      AX_VAR_PUSHVALUE([CPPFLAGS],[$CPPFLAGS $cudainc])
      AX_VAR_PUSHVALUE([CXXFLAGS])
      AX_VAR_PUSHVALUE([LDFLAGS],[$LDFLAGS $cudalib])
      AX_VAR_PUSHVALUE([LIBS],[])
  
      # Check if cuda.h header file exists and compiles
      AC_CHECK_HEADER([cuda.h], [cuda=yes],[cuda=no])

      # Look for cudaMemcpy function in libcudart.so library
      AS_IF([test x$cuda = xyes],[
        AC_SEARCH_LIBS([cudaMemcpy],
                       [cudart],
                       [cuda=yes],
                       [cuda=no])
      ])dnl

      # Look for cublasDrotmg function in libcublas.so library
      AS_IF([test x$cuda = xyes],[
# Note: -lcudart might not be necessary, as it is already included in LIBS
        AC_SEARCH_LIBS([cublasDrotmg],
                       [cublas],
                       [cuda=yes],
                       [cuda=no])
      ])dnl

      # Look for cusparseDrotmg function in libcublas.so library
      AS_IF([test x$cuda = xyes],[
# Note: -lcudart might not be necessary, as it is already included in LIBS
        AC_SEARCH_LIBS([cusparseDcsrmm],
                       [cusparse],
                       [cuda=yes],
                       [cuda=no])
      ])dnl

      cudalibs="$LIBS"
  
      AX_VAR_POPVALUE([CPPFLAGS])
      AX_VAR_POPVALUE([CXXFLAGS])
      AX_VAR_POPVALUE([LDFLAGS])
      AX_VAR_POPVALUE([LIBS])

      AC_LANG_POP([C++])
  
      AS_IF([test x$user_requested = xyes -a x$cuda != xyes],[
          AC_MSG_ERROR([
------------------------------
CUDA path was not correctly specified. 
Please, check that the provided directories are correct.
------------------------------])
      ])dnl
  
		AS_IF([test x$cuda = xyes],[
          AC_CACHE_CHECK([CUDA API version],[ac_cv_cuda_version],
            [
              ac_cv_cuda_version=$(grep 'define CUDA_VERSION' "$cuda_h")
              ac_cv_cuda_version=$(expr "x$ac_cv_cuda_version" : 'x#define CUDA_VERSION \(@<:@0-9@:>@*\)')
            ])
		])dnl

      AS_IF([test x$user_requested = xyes],[
        AS_IF([test "x$ac_cv_cuda_version" == "x" -o "$ac_cv_cuda_version" -lt 5000],[
          AC_MSG_ERROR([
------------------------------
Version of the provided CUDA package is too old.
CUDA 5 or greater is required.
------------------------------])
        ])dnl
      ])dnl

  ])dnl with_cuda
  
  AS_IF([test x$cuda = xyes],[
      ARCHITECTURES="$ARCHITECTURES gpu"
  
      AC_DEFINE_UNQUOTED([NANOS_CUDA_VERSION],[$cuda_version],[API version of the CUDA package specified by the user])
      AC_DEFINE([GPU_DEV],[],[Indicates the presence of the GPU arch plugin.])
  ], [
  cuda_prefix=
  cudainc=
  cudalib=
  cudalibs=
  ])
  
  AC_SUBST([cuda])
  AC_SUBST([cuda_prefix])
  AC_SUBST([cudainc])
  AC_SUBST([cudalib])
  AC_SUBST([cudalibs])

  AM_CONDITIONAL([GPU_SUPPORT],[test x$cuda = xyes])

  AS_IF([test x$cuda = xyes], [
    AC_SUBST([HAVE_CUDA], [GPU_DEV])
  ], [
    AC_SUBST([HAVE_CUDA], [NO_GPU_DEV])
  ])
])dnl AX_CHECK_CUDA
