#
# SYNOPSIS
#
#   AX_CHECK_GASNET
#
# DESCRIPTION
#
#   Check whether GASNet path to the headers and libraries are correctly specified.
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

AC_DEFUN([AX_CHECK_GASNET],
[
AC_REQUIRE([AX_PTHREAD])
AC_REQUIRE([AX_CHECK_MPI])
AC_PREREQ(2.59)

m4_pattern_allow([AM_Poll])

# It is also possible to specify an MPI installation directory where header and library files should be placed
AC_ARG_WITH(gasnet,
[AS_HELP_STRING([--with-gasnet,--with-gasnet=PATH],
                [search in system directories or specify prefix directory for installed GASNet package.])])
AC_ARG_WITH(gasnet-include,
[AS_HELP_STRING([--with-gasnet-include=PATH],
                [specify directory for installed GASNet include files])])
AC_ARG_WITH(gasnet-lib,
[AS_HELP_STRING([--with-gasnet-lib=PATH],
                [specify directory for the installed GASNet library])])

# If the user specifies --with-gasnet, $with_gasnet value will be 'yes'
#                       --without-gasnet, $with_gasnet value will be 'no'
#                       --with-gasnet=somevalue, $with_gasnet value will be 'somevalue'
if [[[ ! "x$with_gasnet" =~  x(yes|no|)$ ]]]; then
  gasnetinc="$with_gasnet/include"
  if test -f $with_gasnet/lib64; then
    gasnetlib="-L$with_gasnet/lib64 -Wl,-rpath=$with_gasnet/lib64"
  else
    gasnetlib="-L$with_gasnet/lib -Wl,-rpath=$with_gasnet/lib"
  fi
fi

# If the user does not specify --with-gasnet, or
# he uses --without-gasnet, do not check for GASNet support.
if [[[ "x$with_gasnet" =~ x(no|)$ ]]]; then
  gasnet="no"
else
  gasnet="yes"
fi

if test "$with_gasnet_include"; then
  gasnetinc="-I$with_gasnet_include"
fi

if test "$with_gasnet_lib"; then
  gasnetlib="-L$with_gasnet_lib -Wl,-rpath=$with_gasnet_lib"
fi

if [[ "x$gasnet" = "xyes" ]]; then

  AC_MSG_CHECKING([for GASNet conduits])

  AC_LANG_PUSH([C++])

  bak_CXX=$CXX
  bak_CPPFLAGS=$CPPFLAGS
  bak_CXXFLAGS=$CXXFLAGS
  bak_LIBS=$LIBS
  bak_LDFLAGS=$LDFLAGS

  CXXFLAGS="$CXXFLAGS $PTHREAD_CFLAGS"

  # Do not print conduit check results in standard output
  AX_SILENT_MODE(on)

  # Check available GASNet conduits that are supported.
  # Supported conduits: smp, udp, mpi, ibv

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
  _AX_CHECK_GASNET_CONDUIT(mpi,$MPICXX,-lammpi)
  _AX_CHECK_GASNET_CONDUIT(ibv,$MPICXX,-libverbs,-DGASNET_CONDUIT_IBV)

  # Checks done. Disable silent mode again.
  AX_SILENT_MODE(off)

  AS_IF([test "x$gasnet_available_conduits" = x],
            [
             AC_MSG_RESULT([none available])
             AC_MSG_ERROR([
------------------------------
GASNet path was not correctly specified. 
Please, check that the provided directories are correct.
------------------------------])],
            [
             AC_MSG_RESULT([$gasnet_available_conduits])
  ])

  AS_IF([test x$mpi = xyes -a x$gasnet_mpi_available != xyes],[
    AC_MSG_WARN([
------------------------------
MPI was enabled but GASNet MPI conduit was not found
or it was not compatible. If MPI and/or InfiniBand conduits
are required, it is recommended to use the same MPI library
GASNet is linked with. 
------------------------------])
  ])

  CXX=$bak_CXX
  CPPFLAGS=$bak_CPPFLAGS
  CXXFLAGS=$bak_CXXFLAGS
  LIBS=$bak_LIBS
  LDFLAGS=$bak_LDFLAGS

  AC_LANG_POP([C++])

  # We will make use of the parallel variant 'GASNET_PAR'
  # This must be defined before including gasnet.h
  AC_DEFINE([GASNET_PAR],[],[Defines which multithreading support is required to GASNet])

fi # if gasnet

m4_foreach_w([conduit_name],[smp udp mpi ibv],[
  _AX_CONDUIT_SUBST(conduit_name)
])

])dnl AX_CHECK_GASNET

# _AX_CHECK_GASNET_CONDUIT(name [, compiler [, libraries [, preprocessor flags ]]])
# Helper function that checks for the availability of a single GASNet conduit
# Parameters:
# $1 - Conduit name. Expected values: {smp, udp, mpi, ibv}
# $2 - Required compiler. Some conduits must be compiled differently (e.g.: mpi must be compiled with MPI compiler)
# $3 - Library requirements (optional). Special library requirements to link with this conduit.
# $4 - Additional preprocessor flags (optional).
AC_DEFUN([_AX_CHECK_GASNET_CONDUIT],
[
  AS_VAR_PUSHDEF([conduit_available],[gasnet_$1_available])
  AS_VAR_PUSHDEF([conduit_inc],  [gasnet_$1_inc])
  AS_VAR_PUSHDEF([conduit_libs], [gasnet_$1_libs])

  CXX=$2
  conduit_prereq_libs="$3 $PTHREAD_LIBS -lrt"
  conduit_inc="-isystem $gasnetinc -I$gasnetinc/$1-conduit $4"

  CPPFLAGS="${bak_CPPFLAGS} $conduit_inc"
  LDFLAGS="${bak_LDFLAGS} $gasnetlib"
  LIBS=


  # We do not want autoconf to cache header/library checks.
  # We will use different environments for each check.
  unset ac_cv_header_gasnet_h
  unset ac_cv_search_gasnetc_AMPoll

  # Skip checks if the required compiler is not found
  AS_IF([test x$CXX = x ],[
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
    
  AS_IF([test x$conduit_available = xyes],[
    gasnet_available_conduits+="$1 "
    AS_VAR_SET([conduit_libs], ["$conduit_prereq_libs $LIBS"])
  ])

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

  AM_CONDITIONAL([conduit_available],[test x$conduit_available = xyes])
  AC_SUBST([conduit_inc])
  AC_SUBST([conduit_libs])

  AS_VAR_POPDEF([conduit_available])dnl
  AS_VAR_POPDEF([conduit_inc])dnl
  AS_VAR_POPDEF([conduit_libs])dnl

])dnl _AX_CONDUIT_SUBST

