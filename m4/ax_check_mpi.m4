#
# SYNOPSIS
#
#   AX_CHECK_MPI
#
# DESCRIPTION
#
#   Check whether MPI path to the headers and libraries is correctly specified.
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

AC_DEFUN([AX_CHECK_MPI],[
AC_REQUIRE([AC_PROG_CXX])
AC_REQUIRE([AC_ARG_VAR])

# Let the user specify C++ MPI compiler using an environment variable
AC_ARG_VAR(MPICXX,[MPI C++ compiler command])

# It is also possible to specify an MPI installation directory where header and library files should be placed
AC_ARG_WITH(mpi,
[AS_HELP_STRING([--with-mpi,--with-mpi=PATH],
                [search in system directories or specify prefix directory for installed OpenCL package.])])
AC_ARG_WITH(mpi-include,
[AS_HELP_STRING([--with-mpi-include=PATH],
                [specify directory for installed OpenCL include files])])
AC_ARG_WITH(mpi-lib,
[AS_HELP_STRING([--with-mpi-lib=PATH],
                [specify directory for the installed OpenCL library])])

# If the user specifies --with-mpi, $with_mpi value will be 'yes'
#                       --without-mpi, $with_mpi value will be 'no'
#                       --with-mpi=somevalue, $with_mpi value will be 'somevalue'
if [[[ ! "x$with_mpi" =~  x(yes|no|)$ ]]]; then
  mpiinc="-I$with_mpi/include"
  AC_CHECK_FILE([$with_mpi/lib64],
    [mpilib="-L$with_mpi/lib64 -Wl,-rpath=$with_mpi/lib64"],
    [mpilib="-L$with_mpi/lib -Wl,-rpath=$with_mpi/lib"])
else if [ "x$with_mpi" = "xno" ]
  mpi="no"
fi

if test $with_mpi_include; then
  mpiinc="-I$with_mpi_include"
fi

if test $with_mpi_lib; then
  mpilib="-L$with_mpi_lib"
fi

if [ "x$mpi" = "xyes"] then

  bak_CXX="$CXX"
  bak_CFLAGS="$CFLAGS"
  bak_CxXFLAGS="$CXXFLAGS"
  bak_CPPFLAGS="$CPPFLAGS"
  bak_LIBS="$LIBS"
  bak_LDFLAGS="$LDFLAGS"
  
  CFLAGS=
  CXXFLAGS=
  CPPFLAGS=
  LIBS=
  LDFLAGS=
  
  AC_LANG_PUSH([C++])
  
  AC_PROG_CXX([$MPICXX mpiicpc mpicxx])
  
  # Check if mpi.h header file exists and compiles
  AC_CHECK_HEADER([mpi.h], [mpi=yes],[mpi=no])
  
  # Check if the provided MPI implementation is Intel MPI
  # Multithread support will be provided if the flag -mt_mpi is used
  if test x$mpi == xyes; then
    LDFLAGS=-mt_mpi
    AC_CHECK_LIB([mpi_mt],
                   [MPI_Init_thread],
                   [impi=yes], # Intel MPI library detected
                   [LDFLAGS=""]) # Remove -mt_mpi flag from LDFLAGS
  fi
  
  # Look for MPI_Init_thread function in libmpi.so library
  if test x$mpi == xyes; then
    AC_CHECK_LIB([mpicxx],
                   [MPI_Init_thread],
                   [mpi=yes],
                   [mpi=no])
  fi
  
  if test x$mpi != xyes; then
      AC_MSG_ERROR([
------------------------------
MPI path was not correctly specified. 
Please, check that provided directories are correct.
------------------------------])
  fi
  
  if test x$mpi = xyes; then
    AC_CACHE_CHECK([if MPI library supports multithreading],[ac_cv_mpi_mt],
      [AC_RUN_IFELSE(
        [AC_LANG_PROGRAM(
          [
             #ifdef HAVE_MPI_H
                 #include <mpi.h>
             #endif
             #include <stdio.h>
             #include <stdlib.h>
          ],
          [
             /* Initialize MPI library. */
             int err = MPI_SUCCESS;
             int provided;
             err = MPI_Init_thread( NULL, NULL, MPI_THREAD_MULTIPLE, &provided );
             if (err != MPI_SUCCESS) {
                 printf( "MPI_Init_thread failed with error %d\n", err );
                 return 1;
             }
  
             char *mt_level;
             switch( provided ) {
                 case MPI_THREAD_SINGLE:
                 mt_level = "MPI_THREAD_SINGLE";
                 break;
  
                 case MPI_THREAD_FUNNELED:
                 mt_level = "MPI_THREAD_FUNNELED";
                 break;
  
                 case MPI_THREAD_SERIALIZED:
                 mt_level = "MPI_THREAD_SERIALIZED";
                 break;
  
                 case MPI_THREAD_MULTIPLE:
                 mt_level = "MPI_THREAD_MULTIPLE";
                 break;
  
                 default:
                 mt_level = "(invalid value)";
                 break;
             }
  
             FILE* out = fopen("conftest.out","w");
             fprintf(out,"%s\n", mt_level);
             fclose(out);
             
             return 0;
          ])],
        [ac_cv_mpi_mt=$(cat conftest.out)
        ],
        [AC_MSG_FAILURE([
------------------------------
The execution of MPI multithread support test failed
------------------------------])
        ])
      ])
  fi

  if [ "x$ac_cv_mpi_mt" != "xMPI_THREAD_MULTIPLE" ]; then
    AC_MSG_FAILURE([
------------------------------
MPI library specified does not support multithreading.
Please, provide a MPI library that does so.
Max. thread level supported: $ac_cv_mpi_mt
------------------------------])
  fi
  
  mpilibs=$LIBS
  MPICXX=$CXX

  CXX="$bak_CXX"
  CFLAGS="$bak_CFLAGS"
  CXXFLAGS="$bak_CXXFLAGS"
  CPPFLAGS="$bak_CPPFLAGS"
  LIBS="$bak_LIBS"
  LDFLAGS="$bak_LDFLAGS"
  
  AC_LANG_POP([C++])

fi # use mpi

AM_CONDITIONAL([MPI_SUPPORT],[test x$mpi = xyes ])

AC_SUBST([mpi])
AC_SUBST([mpiinc])
AC_SUBST([mpilib])

])dnl AX_CHECK_MPI

