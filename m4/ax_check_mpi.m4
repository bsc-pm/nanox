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
#   AX_CHECK_MPI
#
# DESCRIPTION
#
#   Check whether MPI architecture has been enabled or not.
#   We do not include header/library checks in this macro because
#   we do not want Nanos++ to be bound to a specific MPI implementation.
#   MPI dependent plugin source code will be compiled in the user program
#   build stage.
#

AC_DEFUN([AX_CHECK_MPI],[
AC_BEFORE([AC_PROG_CXX],[$0])

# Let the user specify C++ MPI compiler using an environment variable
# For this test, we will not make use of the CXX variable, as it is
# used for regular C++ source code (not MPI).
AC_ARG_VAR(MPICXX,[MPI C++ compiler command])

# It is also possible to specify an MPI installation directory where header and library files should be placed
# If the user does not specify --with-mpi, or
# he uses --without-mpi, do not check for MPI support.
AC_ARG_WITH(mpi,
[AS_HELP_STRING([--with-mpi,--with-mpi=PATH],
                [search in system directories or specify prefix directory for installed MPI package.])],
  [
    # Check if the user provided a valid directory 
    AS_IF([test -d "$withval"],[
      mpi=yes
      mpi_path_provided=yes
    ],[
      mpi=$withval
      mpi_path_provided=no
    ])dnl
  ],[
    mpi=no
    mpi_path_provided=no
  ])

AC_ARG_WITH(mpi-include,
[AS_HELP_STRING([--with-mpi-include=PATH],
                [specify directory for installed MPI include files])])
AC_ARG_WITH(mpi-lib,
[AS_HELP_STRING([--with-mpi-lib=PATH],
                [specify directory for the installed MPI library])])

# If the user specifies --with-mpi, $with_mpi value will be 'yes'
#                       --without-mpi, $with_mpi value will be 'no'
#                       --with-mpi=somevalue, $with_mpi value will be 'somevalue'
AS_IF([test x$mpi_path_provided = xyes],[
  AS_IF([test -d "$with_mpi/bin64"],
    [mpibin=$with_mpi/bin64],
    [mpibin=$with_mpi/bin])dnl

  mpiinc="-I$with_mpi/include"
  AS_IF([test -d $with_mpi/lib64],
    [mpilib="-L$with_mpi/lib64 -Wl,-rpath,$with_mpi/lib64"],
    [mpilib="-L$with_mpi/lib -Wl,-rpath,$with_mpi/lib"])dnl
])dnl

AS_IF([test $with_mpi_include],[
  mpiinc="-I$with_mpi_include"
])dnl

AS_IF([test $with_mpi_lib],[
  mpilib="-L$with_mpi_lib -Wl,-rpath,$with_mpi_lib"
])dnl

AS_IF([test $mpi = yes],[
  # 1) Save the previous value of autoconf generated variables.
  # They may be used later in the configuration (e.g. AC_OUTPUT)
  # 2) Save some important cached values.
  # This macro performs some checks using a different C++ compiler (mpicxx)
  # This is not very standar-ish, but allows us to avoid autoconf to reuse
  # AC_PROG_CXX checks that were done previously.
  # Of course, we will need to restore them before returning.
  # 3) Empty/set values
  # MPICH_IGNORE_CXX_SEEK and MPICH_SKIP_MPICXX are used to avoid
  # errors when mpi.h is included after stdio.h when compiling C++ code
  # It only applies, however, to mpich implementations
  # Some exceptions:
  #  - Dont unset CPPFLAGS, CXXFLAGS and LDFLAGS. Respect additional user provided flags
  AX_VAR_PUSHVALUE([CPPFLAGS],[$CPPFLAGS $mpiinc -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX])
  AX_VAR_PUSHVALUE([CXXFLAGS])
  AX_VAR_PUSHVALUE([LDFLAGS],[$LDFLAGS $mpilib])
  AX_VAR_PUSHVALUE([LIBS])

  # For cached values and tools, it implies unsetting the variable itself,
  # or it will skip their corresponding checks.
  AX_VAR_PUSHVALUE([CXXCPP],[])
  AX_VAR_PUSHVALUE([CXX],[])
  AX_VAR_PUSHVALUE([ac_cv_prog_CXXCPP],[])
  AX_VAR_PUSHVALUE([ac_cv_prog_CXX],[])
  AX_VAR_PUSHVALUE([ac_cv_prog_ac_ct_CXX],[])
  AX_VAR_PUSHVALUE([ac_cv_cxx_compiler_gnu],[])
  AX_VAR_PUSHVALUE([ac_cv_prog_cxx_g],[])
  AX_VAR_PUSHVALUE([am_cv_CXX_dependencies_compiler_type],[])
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
  AC_PROG_CXX([$MPICXX $mpibin/mpiicpc $mpibin/mpicxx $mpibin/mpic++ mpiicpc mpicxx mpic++])
  AC_PROG_CXXCPP()

  # Check if mpi.h and mpicxx.h header files exists and compiles
  AC_CHECK_HEADERS([mpi.h], [mpi=yes],[mpi=no])
  
  # Check if the provided MPI implementation is Intel MPI
  # Multithread support will be provided if the flag -mt_mpi is used
  # or if we link against libmpi_mt library.
  # Newer versions of Intel MPI favor the option -link_mpi=opt_mt,
  # meanwhile -mt_mpi option has been deprecated.
  AX_CHECK_LINK_FLAG([-link_mpi=opt_mt],[
    AX_APPEND_FLAG([ -link_mpi=opt_mt],[LDFLAGS])
  ],[
    AX_CHECK_LINK_FLAG([-mt_mpi],[
      AX_APPEND_FLAG([ -mt_mpi],[LDFLAGS])
    ])
  ])
  
  # Look for MPI_Init_thread function in libmpi_mt, libmpi or libmpich libraries
  AS_IF([test x$mpi == xyes],[
    AC_SEARCH_LIBS([MPI_Init_thread],
                   [mpi_mt mpich mpi],
                   [mpi=yes;break],
                   [mpi=no])dnl
  ])dnl

  # Look for MPI::Comm::Comm() function in libmpicxx, libmpi_cxx or libmpichcxx libraries
  AS_IF([test x$mpi == xyes],[
    AC_SEARCH_LIBS([_ZN3MPI4Comm10DisconnectEv],
                   [mpichcxx mpi_cxx],
                   [mpi=yes;break],
                   [mpi=no])dnl
  ])dnl
  
  # If one of the previous tests were not satisfied, exit with an error message.
  AS_IF([test x$mpi = xyes],[
    ARCHITECTURES="$ARCHITECTURES mpi"
  ],[
      AC_MSG_ERROR([
------------------------------
MPI path was not correctly specified. 
Please, check that provided directories are correct.
------------------------------])
  ])dnl

  # Check that the MPI library supports multithreading (MPI_THREAD_MULTIPLE)
  AS_IF([test x$mpi = xyes],[
    AC_CACHE_CHECK([MPI library multithreading support],[ac_cv_mpi_mt],
      [AC_RUN_IFELSE(
        [AC_LANG_PROGRAM(
          [
             #include <string>

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
  
             std::string mt_support;
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
             fprintf(out,"%s\n", mt_support.c_str());
             fclose(out);
             
             return 0;
          ])],
        [ac_cv_mpi_mt=$(cat conftest.out)
        ],
        [AC_MSG_FAILURE([
------------------------------
The execution of MPI multithread support test failed
------------------------------])
        ],
        [
          # Cross compilation mode
          ac_cv_mpi_mt=skip
          break;
        ])
      ])
  ])dnl

  AS_IF([test $ac_cv_mpi_mt = skip],[
    AC_MSG_WARN([
------------------------------
Multithreading support check was not done
because cross-compilation mode was detected.
------------------------------])
  ],[
    AS_IF([test $ac_cv_mpi_mt != concurrent],[
      AC_MSG_FAILURE([
------------------------------
MPI library specified does not support multithreading.
Please, provide a MPI library that does so.
Maximun multithread level supported: $ac_cv_mpi_mt
------------------------------])
    ])dnl
  ])dnl

  AS_CASE([$LIBS],
    [*"-lmpichcxx "*"-lmpi_mt"*],[mpi_implementation=intel],
    [*"-lmpichcxx "*"-lmpich"*], [mpi_implementation=mpich],
    [*"-lmpicxx "*"-lmpi"*],     [mpi_implementation=openmpi],
    [mpi_implementation=none]
  )
  AC_DEFINE([MPICH_IGNORE_CXX_SEEK],[],[Ignore cxx seek errors when including mpi.h in C++])
  AC_DEFINE_UNQUOTED([MPI_IMPLEMENTATION],[$mpi_implementation],
    [Identifies which MPI implementation is being used. Supported values: intel, mpich, openmpi])

  MPICXX="$CXX"
  mpilib="$LDFLAGS"
  mpilibs="$LIBS"

  # Restore variables to its original state
  AX_VAR_POPVALUE([CPPFLAGS])
  AX_VAR_POPVALUE([CXXFLAGS])
  AX_VAR_POPVALUE([LDFLAGS])
  AX_VAR_POPVALUE([LIBS])

  AX_VAR_POPVALUE([CXXCPP])
  AX_VAR_POPVALUE([CXX])
  AX_VAR_POPVALUE([ac_cv_prog_CXXCPP])
  AX_VAR_POPVALUE([ac_cv_prog_CXX])
  AX_VAR_POPVALUE([ac_cv_prog_ac_ct_CXX])
  AX_VAR_POPVALUE([ac_cv_cxx_compiler_gnu])
  AX_VAR_POPVALUE([ac_cv_prog_cxx_g])
  AX_VAR_POPVALUE([am_cv_CXX_dependencies_compiler_type])
  
  AC_LANG_POP([C++])

])dnl use mpi

AM_CONDITIONAL([MPI_SUPPORT],[test x$mpi = xyes ])

AC_SUBST([MPICXX])
AC_SUBST([mpiinc])
AC_SUBST([mpilib])
AC_SUBST([mpilibs])

])dnl AX_CHECK_MPI

