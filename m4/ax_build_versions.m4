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
#   AX_BUILD_VERSIONS
#   Provides enable/disable configuration flags for performance, debug, instrumentation
#   and instrumentation-debug build versions.
#
# DESCRIPTION
#
#   Check whether OpenCL path to the headers and libraries are correctly specified.
#   Also checks that the library version is OpenCL 1.1 or greater.
#

AC_DEFUN([AX_BUILD_VERSIONS],[

VERSIONS=

# Preprocessor, compiler flags and default availability
# Note: I have to put a prefix "is_" to avoid that
# instrumentation_debug_default can be expanded as
# instrumentation_ + debug_default, so that if
# debug_default is defined to yes,
# instrumentation_debug_default will be expanded as instrumentation_yes

# Performance
m4_define(is_performance_default,yes)
performance_CPPFLAGS=
performance_CXXFLAGS="-O3"

# Debug
m4_define(is_debug_default,yes)
debug_CPPFLAGS="-DNANOS_DEBUG_ENABLED"
debug_CXXFLAGS="-O0 -g2 $no_inline_flag"

# Instrumentation
m4_define(is_instrumentation_default,yes)
instrumentation_CPPFLAGS="-DNANOS_INSTRUMENTATION_ENABLED"
instrumentation_CXXFLAGS="-O3"

# Instrumentation-debug
m4_define(is_instrumentation_debug_default,no)
instrumentation_debug_CPPFLAGS="-DNANOS_DEBUG_ENABLED -DNANOS_INSTRUMENTATION_ENABLED"
instrumentation_debug_CXXFLAGS="-O0 -g2 $no_inline_flag"

m4_foreach([version_name],[[performance], [debug], [instrumentation], [instrumentation-debug]],[
  _ax_enable_version(version_name)
])

AS_IF([test "x$VERSIONS" = x],[
  AC_MSG_ERROR([
------------------------------
At least one version needs to be compiled
------------------------------])
])
AC_SUBST([VERSIONS])

])dnl AX_BUILD_VERSIONS

# Helper function
# Includes enable/disable configuration argument to 'help' dialog
# Checks default values and explicit user configuration
# Substitutes autoconf and automake variables
AC_DEFUN([_ax_enable_version],[
  AS_VAR_PUSHDEF([version_default],[is_$1_default])
  AS_VAR_PUSHDEF([version_enabled],[is_$1_enabled])
  AS_VAR_PUSHDEF([cppflags],[$1_CPPFLAGS])
  AS_VAR_PUSHDEF([cxxflags],[$1_CXXFLAGS])
  AS_VAR_PUSHDEF([config_libs],[nanos_config_libs_$1])

  # Versiondir (e.g. performancedir) is used by libtool
  # to place ltlibraries
  AS_VAR_PUSHDEF([version_dir],[$1dir])

  AC_MSG_CHECKING([if version_name version is enabled])
  AC_ARG_ENABLE($1,
    [m4_if(version_default,yes,
       AS_HELP_STRING(--disable-$1,Disable generation of $1 version),
       AS_HELP_STRING(--enable-$1,Enable generation of $1 version)
    )],
    [version_enabled=${enableval}],
    [version_enabled=]version_default)
  
  AC_MSG_RESULT([$version_enabled])
  
  AS_IF([test x$version_enabled = xyes],[
    VERSIONS+="$1 "
  ])

  # Generate architecture library list (will be used in core/Makefile.am)
  config_libs=
  for arch in $ARCHITECTURES; do
     AS_IF([test x"$arch" != x"mpi"],[
        AS_VAR_APPEND([config_libs],[" \$(abs_top_builddir)/src/arch/$arch/$1/lib$arch.la"])
     ])
  done
  AS_VAR_APPEND([config_libs],[" \$(abs_top_builddir)/src/arch/$OS/$1/libos.la \$(abs_top_builddir)/src/support/$1/libsupport.la"])
  
  version_dir='${libdir}/$1'
  AM_CONDITIONAL(version_enabled, [test x$version_enabled = xyes])
  AC_SUBST(version_dir)
  AC_SUBST(cppflags)
  AC_SUBST(cxxflags)
  AC_SUBST(config_libs)

  AS_VAR_POPDEF([version_default])
  AS_VAR_POPDEF([version_enabled])
  AS_VAR_POPDEF([version_dir])
  AS_VAR_POPDEF([cppflags])
  AS_VAR_POPDEF([cxxflags])
  AS_VAR_POPDEF([config_libs])
])dnl check_version

