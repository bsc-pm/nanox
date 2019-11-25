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
#   AX_CHECK_HOST_ARCH
#
# DESCRIPTION
#
#   Check host architechture
#

AC_DEFUN([AX_CHECK_HOST_ARCH],[
AC_REQUIRE([AC_CANONICAL_SYSTEM])

# This macro checks compilers and compiler flags
# Must be called after checking compiler programs
AC_BEFORE([AC_PROG_CC],[$0])
AC_BEFORE([AC_PROG_CXX],[$0])

AC_CHECK_SIZEOF([size_t])

# Use these for flags that depend on the host arch
# Note: Should we split this into two cases?
# 1) Check host architecture
# 2) Check host OS
# Second case would be only check if it contains
# linux inside triplet.
ult_support=yes
AS_CASE([$host],
  [x86_64-k1om-linux*|k1om-mpss-linux*],
  [
    # Intel MIC (KNF/ KNC)
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"
    SMP_ARCH=x86-64
    # Run specific checks for mic
    AX_CHECK_MIC
  ],
  [x86_64-*-linux*],
  [
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"
    SMP_ARCH=x86-64
  ],
  [i?86-*-linux*],
  [
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"
    SMP_ARCH=x86
  ],
  [ia64-*-linux*],
  [
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"
    SMP_ARCH=ia64
  ],
  [arm*-*-gnueabihf],
  [
    # ARMv7 with hardware floating point support
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"
    SMP_ARCH=armv71_hf
  ],
  [arm*-*-gnueabi],
  [
    # ARMv7 with software floating point operations
    # Other ARMv7 will fail to unknown disabling the ULT
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"
    SMP_ARCH=armv71
  ],
  [aarch64-*-linux-gnu],
  [
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"
    SMP_ARCH=aarch64
  ],
  [powerpc64-bgq-linux-gnu],
  [
    # BlueGene/Q
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"
    SMP_ARCH=ppc64
    AX_CHECK_BGQ
  ],
  [powerpc64??-*-linux*],
  [
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"

    if test $ac_cv_sizeof_size_t = 8 ; then
        SMP_ARCH=ppc64_v2
    fi
    #FIXME ULT not supported yet
    ult_support=no
  ],
  [powerpc-*-linux* | powerpc64-*-linux*],
  [
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"

    if test $ac_cv_sizeof_size_t = 8 ; then
        SMP_ARCH=ppc64
    else
        SMP_ARCH=ppc32
    fi

    # Check if it is a Cell system
    if cat /proc/cpuinfo | grep Cell &> /dev/null ; then
      spu_valid=yes
      AC_ARG_VAR([SPUCC],[SPU C compiler])
      AC_ARG_VAR([PPUEMBED],[SPU to PPU embedding tool])
      AC_CHECK_PROG( [SPUCC], [spu-cc], spu-cc, no)
      if test x$SPUCC = xno ; then
	AC_MSG_WARN([spu-cc not found. Disabling SPU support])
        spu_valid=no
      else
        AC_CHECK_PROG( [PPUEMBED], [ppu-embedspu], ppu-embedspu, no)
        if test x$PPUEMBED = xno ; then
          AC_MSG_WARN([ppu-embedspu not found. Disabling SPU support])
          spu_valid=no
        else
           AC_ARG_WITH([cellsdk],
              AS_HELP_STRING([--with-cellsdk=dir], [Directory of Cell SDK installation]),
              [
		CELLSDK="${withval}"
              ],
              [
		CELLSDK=/opt/cell/sdk/usr
              ])
           CELLSDK_INC="$CELLSDK/include"
           CELLSDK_LIB="$CELLSDK/lib"

	   AC_SUBST([CELLSDK_INC])
	   AC_SUBST([CELLSDK_LIB])

	   NANOS_CONFIG_LIBDIRS="NANOS_CONFIG_LIBDIRS -L$CELLSDK_LIB"
	   NANOS_CONFIG_LIBS="NANOS_CONFIG_LIBS -lspe2"
        fi
      fi
      if test x$spu_valid = xyes; then
        ARCHITECTURES="$ARCHITECTURES spu"
      fi
    fi
  ],
  [riscv64-*-linux-gnu],
  [
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"
    SMP_ARCH=riscv64
  ],
  [
    # Default
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"
    SMP_ARCH=unknown
    ult_support=no
  ]
)dnl AS_CASE

# User level threads
# System dependent. Architecture support will make a compatibility check.
AC_ARG_ENABLE([ult],
  AS_HELP_STRING([--disable-ult], [Disables user level threads]),
  [enable_ult=${enableval}],
  [enable_ult=yes]
)

AS_IF([test "$ult_support" = yes dnl
         -a "$enable_ult"  = yes],[
  ult_support=yes
  AC_DEFINE([SMP_SUPPORTS_ULT],[],[Indicates support for user level threads])
  AC_SUBST([SMP_SUPPORTS_ULT], [SMP_SUPPORTS_ULT])
],[
  AC_SUBST([SMP_SUPPORTS_ULT], [NO_SMP_SUPPORTS_ULT])
  ult_support=no
])

AC_SUBST([OS])
AC_SUBST([SMP_ARCH])
AM_CONDITIONAL([SMP_SUPPORTS_ULT],[test "$ult_support" = yes])

]) dnl AX_CHECK_HOST_ARCH
