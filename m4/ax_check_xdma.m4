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
#   AX_CHECK_XDMA
#
# DESCRIPTION
#
#   Check whether a valid Zynq Xilinx DMA library is available, and the path to the headers 
#   and libraries are correctly specified.
#

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

