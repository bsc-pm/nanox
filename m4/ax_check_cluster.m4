#
# SYNOPSIS
#
#   AX_CHECK_CLUSTER (May be changed by gasnet)
#
# DESCRIPTION
#
#   Check whether OpenCL path to the headers and libraries are correctly specified.
#   Also checks that the library version is OpenCL 1.1 or greater.
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

AC_DEFUN([AX_CHECK_CLUSTER],
[
AC_PREREQ(2.59)

# Check if user specifically requested Cluster architecture
AC_MSG_CHECKING([if Cluster architecture was requested])
AC_ARG_ENABLE([cluster],
    AS_HELP_STRING([--enable-cluster], [Enables cluster architecture]),
    [
        if test x$enableval = xyes -o x$enableval = x;
        then
            cluster_enabled="yes"
            AC_MSG_RESULT([yes])
        else if test x$enableval = xno;
            then
                AC_MSG_RESULT([no])
                cluster_enabled="no"
             else
                AC_MSG_ERROR([This option can only be given 'yes' or 'no' values])
             fi
        fi
    ],
    [
        cluster_enabled="no"
        AC_MSG_RESULT([yes])
    ]
)
with_mpi_conduit="no"
with_udp_conduit="no"
with_smp_conduit="no"
with_ibv_conduit="no"
with_mpi_conduit_dbg="no"
with_udp_conduit_dbg="no"
with_smp_conduit_dbg="no"
with_ibv_conduit_dbg="no"
if test x$cluster_enabled = xyes ; then

    m4_include(lx_find_mpi.m4)

    CFLAGS="$CFLAGS -DCLUSTER_DEV"
    CXXFLAGS="$CXXFLAGS -DCLUSTER_DEV"
    ARCHITECTURES="$ARCHITECTURES cluster"

    AC_ARG_WITH([gasnet],
          AS_HELP_STRING([--with-gasnet=dir], [Directory of GASNet installation]),
          [
    	GASNET_DIR="${withval}"
          ],
          [
    	GASNET_DIR=/opt/gasnet
          ])

    AC_ARG_WITH([gasnet-dbg],
          AS_HELP_STRING([--with-gasnet-dbg=dir], [Directory of GASNet installation with debug options enabled]),
          [
    	GASNET_DBG_DIR="${withval}"
          ],
          [
    	GASNET_DBG_DIR=$GASNET_DIR
          ])

    AC_SUBST([GASNET_DIR])
    AC_SUBST([GASNET_DBG_DIR])
    AC_SUBST([MPI_CLDFLAGS])
    CONDUITS=`ls -d $GASNET_DIR/include/*-conduit | sed -e "s#$GASNET_DIR/include/##" | cut -d - -f 1`
    for this_conduit in $CONDUITS ; do
        if test x$this_conduit = xmpi ; then
            with_mpi_conduit="yes"
            #AC_LANG_PUSH([C++])
            export MP_CXX=gcc
            LX_FIND_MPI()
            #AC_LANG_POP([C++])
            #echo have_C_mpi is $have_C_mpi : MPICC is $MPICC / MPI_CFLAGS is $MPI_CFLAGS / MPI_CLDFLAGS is $MPI_CLDFLAGS
            #echo have_CXX_mpi is $have_CXX_mpi : MPICXX is $MPICXX / MPI_CXXFLAGS is $MPI_CXXFLAGS / MPI_CXXLDFLAGS is $MPI_CXXLDFLAGS
        fi

        if test x$this_conduit = xudp ; then
            with_udp_conduit="yes"
        fi

        if test x$this_conduit = xsmp ; then
            with_smp_conduit="yes"
        fi

        if test x$this_conduit = xibv ; then
            with_ibv_conduit="yes"
        fi
    done
    if test x$GASNET_DIR != x$GASNET_DBG_DIR ; then
       CONDUITS_DBG=`ls -d $GASNET_DBG_DIR/include/*-conduit | sed -e "s#$GASNET_DBG_DIR/include/##" | cut -d - -f 1`
       for this_conduit in $CONDUITS_DBG ; do
           if test x$this_conduit = xmpi ; then
               with_mpi_conduit_dbg="yes"
               #AC_LANG_PUSH([C++])
               export MP_CXX=gcc
               LX_FIND_MPI()
               #AC_LANG_POP([C++])
               #echo have_C_mpi is $have_C_mpi : MPICC is $MPICC / MPI_CFLAGS is $MPI_CFLAGS / MPI_CLDFLAGS is $MPI_CLDFLAGS
               #echo have_CXX_mpi is $have_CXX_mpi : MPICXX is $MPICXX / MPI_CXXFLAGS is $MPI_CXXFLAGS / MPI_CXXLDFLAGS is $MPI_CXXLDFLAGS
           fi

           if test x$this_conduit = xudp ; then
               with_udp_conduit_dbg="yes"
           fi

           if test x$this_conduit = xsmp ; then
               with_smp_conduit_dbg="yes"
           fi

           if test x$this_conduit = xibv ; then
               with_ibv_conduit_dbg="yes"
           fi
       done
    fi
fi
AM_CONDITIONAL([CONDUIT_MPI], [test x$with_mpi_conduit = xyes])
AM_CONDITIONAL([CONDUIT_UDP], [test x$with_udp_conduit = xyes])
AM_CONDITIONAL([CONDUIT_SMP], [test x$with_smp_conduit = xyes])
AM_CONDITIONAL([CONDUIT_IBV], [test x$with_ibv_conduit = xyes])
AM_CONDITIONAL([CONDUIT_DBG_MPI], [test x$with_mpi_conduit_dbg = xyes])
AM_CONDITIONAL([CONDUIT_DBG_UDP], [test x$with_udp_conduit_dbg = xyes])
AM_CONDITIONAL([CONDUIT_DBG_SMP], [test x$with_smp_conduit_dbg = xyes])
AM_CONDITIONAL([CONDUIT_DBG_IBV], [test x$with_ibv_conduit_dbg = xyes])
