#
# SYNOPSIS
#
#   AX_CHECK_GASNET
#
# DESCRIPTION
#
#   Check whether Dynamic Load Balancing (DLB) path to the headers and libraries are correctly specified.
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

AC_DEFUN([AX_CHECK_DLB],
[
AC_REQUIRE([AX_PTHREAD])

# It is also possible to specify an MPI installation directory where header and library files should be placed
AC_ARG_WITH(dlb,
  [AS_HELP_STRING([--with-dlb,--with-dlb=PATH],
                  [search in system directories or specify prefix directory for installed GASNet package.])],
  [
    # Check if the user provided a valid PATH
    AS_IF([test -d "$withval"],[
      dlb_check=yes
      dlb_path_provided=yes
    ],[
      dlb_check=$withval
      dlb_path_provided=no
    ])dnl
  ],[
    dlb_check=no
    dlb_path_provided=no
  ])

AC_ARG_WITH(dlb-include,
[AS_HELP_STRING([--with-dlb-include=PATH],
                [specify directory for installed GASNet include files])])
AC_ARG_WITH(dlb-lib,
[AS_HELP_STRING([--with-dlb-lib=PATH],
                [specify directory for the installed GASNet library])])

# If the user specifies --with-dlb, $with_dlb value will be 'yes'
#                       --without-dlb, $with_dlb value will be 'no'
#                       --with-dlb=somevalue, $with_dlb value will be 'somevalue'
AS_IF([test x$dlb_path_provided = xyes],[
  dlbinc="-I$with_dlb/include"
  dlblib="-L$with_dlb/lib -Wl,-rpath,$with_dlb/lib"
])dnl

AS_IF([test "$with_dlb_include"],[
  dlbinc="-I$with_dlb_include"
])dnl

AS_IF([test "$with_dlb_lib"],[
  dlblib="-L$with_dlb_lib -Wl,-rpath,$with_dlb_lib"
])dnl

AS_IF([test "$dlb_check" = yes],[

  AC_LANG_PUSH([C++])

  AX_VAR_PUSHVALUE([CPPFLAGS])
  AX_VAR_PUSHVALUE([CXXFLAGS])
  AX_VAR_PUSHVALUE([LDLAGS])
  AX_VAR_PUSHVALUE([LIBS],[])

  AX_APPEND_FLAG([$dlbinc],[CPPFLAGS])
  AX_APPEND_FLAG([$PTHREAD_CFLAGS],[CXXFLAGS])
  AX_APPEND_FLAG([$dlblib],[LDFLAGS])

  # Look for a valid header file
  AC_CHECK_HEADERS([DLB_interface.h],
                [dlb_check=yes],
                [dlb_check=no])

  # Check all DLB library versions: dlb dlb_dbg dlb_instr dlb_instr_dbg
  m4_foreach([version],[[dlb],[dlb_dbg],[dlb_instr],[dlb_instr_dbg]],[
    _AX_CHECK_DLB_LIB_VERSION(version)
  ])

  AS_IF([test "$dlb_check" = no],
    [
      AC_MSG_ERROR([
------------------------------
DLB path was not correctly specified. 
Please, check that the provided directories are correct.
------------------------------])
    ])dnl

  AX_VAR_POPVALUE([CPPFLAGS])
  AX_VAR_POPVALUE([CXXFLAGS])
  AX_VAR_POPVALUE([LDLAGS])
  AX_VAR_POPVALUE([LIBS])

  AC_LANG_POP([C++])

  # Enable support.
  AC_DEFINE([DLB],[],[Enables DLB support.])

])dnl if dlb

AM_CONDITIONAL([DLB], [test "$dlb_check" = yes])
AC_SUBST([dlbinc])
AC_SUBST([dlblib])
AC_SUBST([dlb])
AC_SUBST([dlb_dbg])
AC_SUBST([dlb_instr])
AC_SUBST([dlb_instr_dbg])

AS_IF([test "$dlb_check" = yes], [
  AC_SUBST([HAVE_DLB], [DLB])
], [
  AC_SUBST([HAVE_DLB], [NO_DLB])
])

])dnl AX_CHECK_DLB

# _AX_CHECK_DLB_LIB_VERSION(version)
# Helper function that checks for the availability of a single DLB library version
# Parameters:
# $1 - Version name: dlb dlb_dbg dlb_instr dlb_instr_dbg
AC_DEFUN([_AX_CHECK_DLB_LIB_VERSION],
[
  AS_VAR_PUSHDEF([lib_name],[$1])

  LIBS=

  AS_IF([test "$dlb_check" = yes],[
    AC_SEARCH_LIBS([DLB_Init],
      [lib_name],
      [dlb_check=yes],
      [dlb_check=no])
  ])

  unset ac_cv_search_DLB_Init

  AS_VAR_SET([lib_name],[$LIBS])
  
  AS_VAR_POPDEF([lib_name])dnl

])dnl _AX_CHECK_CONDUIT
