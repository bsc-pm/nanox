#
# SYNOPSIS
#
#   AX_CHECK_XDMA
#
# DESCRIPTION
#
#   Check whether a valid Zynq Xilinx DMA library is available, and the path to the headers 
#   and libraries are correctly specified.
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

AC_DEFUN([AX_CHECK_XDMA],
[
#Check if XDMA lib is installed.
AC_ARG_WITH(xdma,
  [AS_HELP_STRING([--with-xdma,--with-xdma=PATH],
                [search in system directories or specify prefix directory for installed XDMA package.])],
  [
    # Check if the user provided a valid PATH
    AS_IF([test -d "$withval"],[
      xdma=yes
      xdma_path_provided=yes
    ],[
      xdma=$withval
      xdma_path_provided=no
    ])dnl
  ],[
    # Default: check if xdma is available
    xdma=no
    xdma_path_provided=no
  ])

# If the user specifies --with-xdma, $with_xdma value will be 'yes'
#                       --without-xdma, $with_xdma value will be 'no'
#                       --with-xdma=somevalue, $with_xdma value will be 'somevalue'
AS_IF([test "$xdma_path_provided" = yes],[
  xdmainc="-I$with_xdma/include"
  AS_IF([test -d $with_xdma/lib64],
    [xdmalib="-L$with_xdma/lib64 -Wl,-rpath,$with_xdma/lib64"],
    [xdmalib="-L$with_xdma/lib -Wl,-rpath,$with_xdma/lib"])dnl
])dnl

AS_IF([test "$xdma" = yes],[
  AX_VAR_PUSHVALUE([CPPFLAGS],[$CPPFLAGS $xdmainc])
  AX_VAR_PUSHVALUE([CXXFLAGS])
  AX_VAR_PUSHVALUE([LDFLAGS],[$LDFLAGS $xdmalib])
  AX_VAR_PUSHVALUE([LIBS],[])

  # Check for header
  AC_CHECK_HEADERS([libxdma.h],
    [xdma=yes],
    [xdma=no])

  # Look for xdma_init function in libxdma
  AS_IF([test "$xdma" = yes],[
      AC_SEARCH_LIBS([xdmaOpen],
                [xdma],
                [xdma=yes],
                [xdma=no])
  ])dnl

  
  xdmalibs="$LIBS"

  AX_VAR_POPVALUE([CPPFLAGS])
  AX_VAR_POPVALUE([CXXFLAGS])
  AX_VAR_POPVALUE([LDFLAGS])
  AX_VAR_POPVALUE([LIBS])
  
  AS_IF([test "$xdma" = yes],[
    ARCHITECTURES="$ARCHITECTURES fpga"
    AC_DEFINE([FPGA_DEV],[],[Enables FPGA support])
  ])dnl
])dnl

AC_SUBST([xdmainc])
AC_SUBST([xdmalib])
AC_SUBST([xdmalibs])
AM_CONDITIONAL([XDMA_SUPPORT],[test "$xdma" = yes])

AS_IF([test "$xdma" = yes], [
   AC_SUBST([HAVE_XDMA], [FPGA_DEV])
], [
   AC_SUBST([HAVE_XDMA], [NO_FPGA_DEV])
])

])dnl AX_CHECK_XDMA

