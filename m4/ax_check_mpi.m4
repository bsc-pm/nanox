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
#

AC_DEFUN([AX_CHECK_MPI],[
AC_BEFORE([AC_PROG_CXX],[$0])

# Let the user specify C++ MPI compiler using an environment variable
# For this test, we will not make use of the CXX variable, as it is
# used for regular C++ source code (not MPI).
AC_ARG_VAR(MPICXX,[MPI C++ compiler command])

# It is also possible to specify an MPI installation directory where header and library files should be placed
AC_ARG_WITH(mpi,
[AS_HELP_STRING([--with-mpi,--with-mpi=PATH],
                [search in system directories or specify prefix directory for installed MPI package.])])
AC_ARG_WITH(mpi-include,
[AS_HELP_STRING([--with-mpi-include=PATH],
                [specify directory for installed MPI include files])])
AC_ARG_WITH(mpi-lib,
[AS_HELP_STRING([--with-mpi-lib=PATH],
                [specify directory for the installed MPI library])])

# If the user specifies --with-mpi, $with_mpi value will be 'yes'
#                       --without-mpi, $with_mpi value will be 'no'
#                       --with-mpi=somevalue, $with_mpi value will be 'somevalue'
if [[[ ! "x$with_mpi" =~  x(yes|no|)$ ]]]; then
  if test -f "$with_mpi/bin64"; then
    mpibin=$with_mpi/bin64
  else
    mpibin=$with_mpi/bin
  fi

  mpiinc="-I$with_mpi/include"
  AC_CHECK_FILE([$with_mpi/lib64],
    [mpilib="-L$with_mpi/lib64 -Wl,-rpath=$with_mpi/lib64"],
    [mpilib="-L$with_mpi/lib -Wl,-rpath=$with_mpi/lib"])
fi

# If the user does not specify --with-mpi, or
# he uses --without-mpi, do not check for MPI support.
if [[[ "x$with_mpi" =~ x(no|)$ ]]]; then
  mpi="no"
else
  mpi="yes"
fi

if test $with_mpi_include; then
  mpiinc="-I$with_mpi_include"
fi

if test $with_mpi_lib; then
  mpilib="-L$with_mpi_lib -Wl,-rpath=$with_mpi_lib"
fi

if test $mpi = yes; then

  # Save the previous value of autoconf generated variables.
  # They may be used later in the configuration (e.g. AC_OUTPUT)
  save_CXX=$CXX
  save_CxXFLAGS=$CXXFLAGS
  save_CPPFLAGS=$CPPFLAGS
  save_LIBS=$LIBS
  save_LDFLAGS=$LDFLAGS

  # Also save some important cached values.
  # This macro performs some checks using a different C++ compiler (mpicxx)
  # This is not very standar-ish, but allows us to avoid autoconf to reuse
  # AC_PROG_CXX checks that were done previously.
  # Of course, we will need to restore them before returning.
  save_CXXCPP=$CXXCPP
  save_CXX=$CXX
  save_ac_cv_prog_CXXCPP=$ac_cv_prog_CXXCPP
  save_ac_cv_prog_CXX=$ac_cv_prog_CXX
  save_ac_cv_prog_ac_ct_CXX=$ac_cv_prog_ac_ct_CXX
  save_ac_cv_cxx_compiler_gnu=$ac_cv_cxx_compiler_gnu
  save_ac_cv_prog_cxx_g=$ac_cv_prog_cxx_g
  save_am_cv_CXX_dependencies_compiler_type=$am_cv_CXX_dependencies_compiler_type
  
  # Empty/set values
  # MPICH_IGNORE_CXX_SEEK and MPICH_SKIP_MPICXX are used to avoid
  # errors when mpi.h is included after stdio.h when compiling C++ code
  # It only applies, however, to mpich implementations
  CPPFLAGS="$mpiinc -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX"
  CXXFLAGS=
  LIBS=
  LDFLAGS=$mpilibs

  # For cached values, it implies unsetting the variable itself, or it will skip
  # the corresponding checks.
  unset CXXCPP
  unset CXX
  unset ac_cv_prog_CXXCPP
  unset ac_cv_prog_CXX
  unset ac_cv_prog_ac_ct_CXX
  unset ac_cv_cxx_compiler_gnu
  unset ac_cv_prog_cxx_g
  unset am_cv_CXX_dependencies_compiler_type
  
  AC_LANG_PUSH([C++])
  
  # Check for a valid MPI compiler
  AC_PROG_CXX([$MPICXX mpiicpc mpicxx])
  AC_PROG_CXXCPP()

  # Check if mpi.h and mpicxx.h header files exists and compiles
  AC_CHECK_HEADERS([mpi.h], [mpi=yes],[mpi=no])
  
  # Check if the provided MPI implementation is Intel MPI
  # Multithread support will be provided if the flag -mt_mpi is used
  if test x$mpi == xyes; then
    LDFLAGS=-mt_mpi
    AC_CHECK_LIB([mpi_mt],
                   [MPI_Init_thread],
                   [impi=yes],   # Intel MPI library detected
                   [LDFLAGS=""]) # This is not IMPI: remove -mt_mpi flag from LDFLAGS
  fi
  
  # Look for MPI_Init_thread function in libmpicxx, libmpi_cxx or libmpichcxx libraries
  if test x$mpi == xyes; then
    AC_SEARCH_LIBS([MPI_Init_thread],
                   [mpicxx mpi_cxx mpichcxx],
                   [mpi=yes;break],
                   [mpi=no])
  fi
  
  # If one of the previous tests were not satisfied, exit with an error message.
  if test x$mpi != xyes; then
      AC_MSG_ERROR([
------------------------------
MPI path was not correctly specified. 
Please, check that provided directories are correct.
------------------------------])
  fi

  # Check that the MPI library supports multithreading (MPI_THREAD_MULTIPLE)
  if test x$mpi = xyes; then
    AC_CACHE_CHECK([MPI library multithreading support],[ac_cv_mpi_mt],
      [AC_RUN_IFELSE(
        [AC_LANG_PROGRAM(
          [
             #include <stdio.h>
             #include <stdlib.h>
             #include <unistd.h>
             // Include mpi.h after stdio.h
             // Avoid MPICH 'SEEK_SET already defined' error
             #ifdef HAVE_MPI_H
               #include <mpi.h>
             #endif
          ],
          [
             // This library likes printing error messages through stdout
             // redirect them to stderr
             close(1);
             dup(2);

             /* Initialize MPI library. */
             int err = MPI_SUCCESS;
             int provided;
             err = MPI_Init_thread( NULL, NULL, MPI_THREAD_MULTIPLE, &provided );
             if (err != MPI_SUCCESS) {
                 fprintf(stderr, "MPI_Init_thread failed with error %d\n", err );
                 return 1;
             }
  
             char *mt_support;
             switch( provided ) {
                 case MPI_THREAD_SINGLE:
                 mt_support = "none";
                 break;
  
                 case MPI_THREAD_FUNNELED:
                 mt_support = "funneled";
                 break;
  
                 case MPI_THREAD_SERIALIZED:
                 mt_support = "serialized";
                 break;
  
                 case MPI_THREAD_MULTIPLE:
                 mt_support = "concurrent";
                 break;
  
                 default:
                 mt_support = "(invalid value)";
                 break;
             }
  
             FILE* out = fopen("conftest.out","w");
             fprintf(out,"%s\n", mt_support);
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

  if test $ac_cv_mpi_mt != concurrent; then
    AC_MSG_FAILURE([
------------------------------
MPI library specified does not support multithreading.
Please, provide a MPI library that does so.
Maximun multithread level supported: $ac_cv_mpi_mt
------------------------------])
  fi
  
  mpilib=$LIBS
  mpildflags=$LDFLAGS
  MPICXX=$CXX

  # Restore variables to its original state
  CXX=$save_CXX
  CXXFLAGS=$save_CxXFLAGS
  CPPFLAGS=$save_CPPFLAGS
  LIBS=$save_LIBS
  LDFLAGS=$save_LDFLAGS
  
  CXXCPP=$save_CXXCPP
  CXX=$save_CXX
  ac_cv_prog_CXXCPP=$save_ac_cv_prog_CXXCPP
  ac_cv_prog_CXX=$save_ac_cv_prog_CXX
  ac_cv_prog_ac_ct_CXX=$save_ac_cv_prog_ac_ct_CXX
  ac_cv_cxx_compiler_gnu=$save_ac_cv_cxx_compiler_gnu
  ac_cv_prog_cxx_g=$save_ac_cv_prog_cxx_g
  am_cv_CXX_dependencies_compiler_type=$save_am_cv_CXX_dependencies_compiler_type
  
  AC_LANG_POP([C++])

fi

AM_CONDITIONAL([MPI_SUPPORT],[test x$mpi = xyes ])

AC_SUBST([MPICXX])
AC_SUBST([mpiinc])
AC_SUBST([mpilib])
AC_SUBST([mpildflags])

])dnl AX_CHECK_MPI

