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
#   AX_CHECK_GASNET
#
# DESCRIPTION
#
#   Check whether GASNet path to the headers and libraries are correctly specified.
#

AC_DEFUN([AX_CHECK_GASNET],
[
AC_REQUIRE([AX_PTHREAD])
AC_REQUIRE([AX_CHECK_MPI])

m4_pattern_allow([AM_Poll])

# It is also possible to specify an MPI installation directory where header and library files should be placed
AC_ARG_WITH(gasnet,
  [AS_HELP_STRING([--with-gasnet,--with-gasnet=PATH],
                  [search in system directories or specify prefix directory for installed GASNet package.])],
  [
    # Check if the user provided a valid PATH
    AS_IF([test -d "$withval"],[
      gasnet=yes
      gasnet_path_provided=yes
    ],[
      gasnet=$withval
      gasnet_path_provided=no
    ])dnl
  ],[
    gasnet=no
    gasnet_path_provided=no
  ])

AC_ARG_WITH(gasnet-include,
[AS_HELP_STRING([--with-gasnet-include=PATH],
                [specify directory for installed GASNet include files])])
AC_ARG_WITH(gasnet-lib,
[AS_HELP_STRING([--with-gasnet-lib=PATH],
                [specify directory for the installed GASNet library])])

# If the user specifies --with-gasnet, $with_gasnet value will be 'yes'
#                       --without-gasnet, $with_gasnet value will be 'no'
#                       --with-gasnet=somevalue, $with_gasnet value will be 'somevalue'
AS_IF([test x$gasnet_path_provided = xyes],[
  gasnetinc="$with_gasnet/include"
  AS_IF([test -d "$with_gasnet/lib64"],
    [gasnetlib="-L$with_gasnet/lib64 -Wl,-rpath,$with_gasnet/lib64"],
    [gasnetlib="-L$with_gasnet/lib -Wl,-rpath,$with_gasnet/lib"])dnl
])dnl

AS_IF([test "$with_gasnet_include"],[
  gasnetinc="$with_gasnet_include"
])dnl

AS_IF([test "$with_gasnet_lib"],[
  gasnetlib="-L$with_gasnet_lib -Wl,-rpath,$with_gasnet_lib"
])dnl

AS_IF([test "x$gasnet" = xyes],[

  AC_MSG_CHECKING([for GASNet conduits])

  AC_LANG_PUSH([C++])

  AX_VAR_PUSHVALUE([CXX])
  AX_VAR_PUSHVALUE([CPPFLAGS],[$CPPFLAGS -DGASNET_PAR])
  AX_VAR_PUSHVALUE([CXXFLAGS],[$CXXFLAGS $PTHREAD_CFLAGS])
  AX_VAR_PUSHVALUE([LIBS],[])

  # Do not print conduit check results in standard output
  #AX_SILENT_MODE(on)

  # Check available GASNet conduits that are supported.
  # Supported conduits: smp, udp, mpi, ibv, mxm, aries

  # Special requirements for supported conduits
  # SMP: no special requirements
  # UDP: link with libamudp.a
  # MPI: link with libammpi.a
  # IBV: link with infiniband sys libraries
  #      define GASNET_CONDUIT_IBV
  # MPI and IBV conduits require an available MPI
  # compiler

  gasnet_available_conduits=
  _AX_CHECK_GASNET_CONDUIT(smp,$CXX)
  _AX_CHECK_GASNET_CONDUIT(udp,$CXX,-lamudp)
  _AX_CHECK_GASNET_CONDUIT(aries,$CXX,[-L/opt/cray/pmi/5.0.10-1.0000.11050.0.0.ari/lib64 -lpmi -L/opt/cray/ugni/6.0-1.0502.10863.8.29.ari/lib64 -lugni -Wl,--whole-archive,-lhugetlbfs,--no-whole-archive],-DGASNET_CONDUIT_ARIES)

  # set the appropiate LDFLAGS for conduits that require MPI
  AX_VAR_PUSHVALUE([LDFLAGS],[$LDFLAGS $mpilib])

  _AX_CHECK_GASNET_CONDUIT(mpi,$MPICXX,-lammpi $mpilibs)
  _AX_CHECK_GASNET_CONDUIT(ibv,$MPICXX,-libverbs $mpilibs,-DGASNET_CONDUIT_IBV)
  _AX_CHECK_GASNET_CONDUIT(mxm,$MPICXX,-lmxm -L/opt/mellanox/mxm/lib $mpilibs)

  AX_VAR_POPVALUE([LDFLAGS])

  # Checks done. Disable silent mode again.
  #AX_SILENT_MODE(off)

  AS_IF([test "x$gasnet_available_conduits" = x],
    [
      AC_MSG_RESULT([none available])
      AC_MSG_ERROR([
------------------------------
GASNet path was not correctly specified. 
Please, check that the provided directories are correct.
------------------------------])
    ],[
      AC_MSG_RESULT([$gasnet_available_conduits])
      # GASNet conduits found. Enable cluster architecture
      ARCHITECTURES="$ARCHITECTURES cluster"
      AC_DEFINE([CLUSTER_DEV],[],[Indicates the presence of the Cluster arch plugin.])
    ])dnl

  AS_IF([test x$mpi = xyes -a x$gasnet_mpi_available != xyes],[
    AC_MSG_WARN([
------------------------------
MPI was enabled but GASNet MPI conduit was not found
or it was not compatible. If MPI and/or InfiniBand conduits
are required, it is recommended to use the same MPI library
GASNet is linked with. 
------------------------------])
  ])dnl

  AX_VAR_POPVALUE([CXX])
  AX_VAR_POPVALUE([CPPFLAGS])
  AX_VAR_POPVALUE([CXXFLAGS])
  AX_VAR_POPVALUE([LIBS])

  AC_LANG_POP([C++])

  # We will make use of the parallel variant 'GASNET_PAR'
  # This must be defined before including gasnet.h
  AC_DEFINE([GASNET_PAR],[],[Defines which multithreading support is required to GASNet])

])dnl if gasnet

AS_IF([test "x$gasnet_available_conduits" != x],[
  AC_SUBST([HAVE_GASNET], [CLUSTER_DEV])
], [
  AC_SUBST([HAVE_GASNET], [NO_CLUSTER_DEV])
])

m4_foreach_w([conduit_name],[smp udp mpi ibv mxm aries],[
  _AX_CONDUIT_SUBST(conduit_name)
])

])dnl AX_CHECK_GASNET

# _AX_CHECK_GASNET_CONDUIT(name [, compiler [, libraries [, preprocessor flags ]]])
# Helper function that checks for the availability of a single GASNet conduit
# Parameters:
# $1 - Conduit name. Expected values: {smp, udp, mpi, ibv, aries}
# $2 - Required compiler. Some conduits must be compiled differently (e.g.: mpi must be compiled with MPI compiler)
# $3 - Library requirements (optional). Special library requirements to link with this conduit.
# $4 - Additional preprocessor flags (optional).
AC_DEFUN([_AX_CHECK_GASNET_CONDUIT],
[
  AS_VAR_PUSHDEF([conduit_available],[gasnet_$1_available])
  AS_VAR_PUSHDEF([conduit_inc],  [gasnet_$1_inc])
  AS_VAR_PUSHDEF([conduit_libs], [gasnet_$1_libs])

  conduit_prereq_libs="$PTHREAD_LIBS -lrt $3"
  conduit_inc="-isystem $gasnetinc -I$gasnetinc/$1-conduit $4"

  AX_VAR_PUSHVALUE([CXX],[$2])
  AX_VAR_PUSHVALUE([CPPFLAGS],[$CPPFLAGS $conduit_inc])
  AX_VAR_PUSHVALUE([LDFLAGS],[$LDFLAGS $gasnetlib])
  AX_VAR_PUSHVALUE([LIBS],[])

  # We do not want autoconf to cache header/library checks.
  # We will use different environments for each check.
  unset ac_cv_header_gasnet_h
  unset ac_cv_search_gasnetc_AMPoll

  # Skip checks if the required compiler is not found
  AS_IF([test "x$CXX" = x ],[
    conduit_available=no
  ],[
    # Look for a valid header file
    AC_CHECK_HEADERS([gasnet.h],
                  [conduit_available=yes])
    # Check that library exist
    AS_IF([test x$conduit_available = xyes],[
      AC_SEARCH_LIBS([gasnetc_AMPoll],
                  [gasnet-$1-par],
                  [conduit_available=yes],
                  [conduit_available=no],
                  [$conduit_prereq_libs])
    ])
  ])

  
  AS_IF([test "$conduit_available" = yes],[
    AS_VAR_APPEND([gasnet_available_conduits],[" $1"])
    AS_VAR_SET([conduit_libs],["$LIBS $conduit_prereq_libs"])
  ])

  AX_VAR_POPVALUE([CXX])
  AX_VAR_POPVALUE([CPPFLAGS])
  AX_VAR_POPVALUE([LDFLAGS])
  AX_VAR_POPVALUE([LIBS])

  AS_VAR_POPDEF([conduit_available])dnl
  AS_VAR_POPDEF([conduit_inc])dnl
  AS_VAR_POPDEF([conduit_libs])dnl

])dnl _AX_CHECK_CONDUIT

# _AX_CONDUIT_SUBST
# Sets values to automake and autoconf substitution variables
# for an specific GASNet conduit.
# Params:
#   1) conduit name
AC_DEFUN([_AX_CONDUIT_SUBST],[

  AS_VAR_PUSHDEF([conduit_available],[gasnet_$1_available])dnl
  AS_VAR_PUSHDEF([conduit_inc],  [gasnet_$1_inc])dnl
  AS_VAR_PUSHDEF([conduit_libs], [gasnet_$1_libs])dnl

  AM_CONDITIONAL(conduit_available,[test x$conduit_available = xyes])
  AC_SUBST(conduit_inc)
  AC_SUBST(conduit_libs)
  AC_SUBST([gasnetlib])

  AS_VAR_POPDEF([conduit_available])dnl
  AS_VAR_POPDEF([conduit_inc])dnl
  AS_VAR_POPDEF([conduit_libs])dnl

])dnl _AX_CONDUIT_SUBST

