# ===========================================================================
#   http://www.gnu.org/software/autoconf-archive/ax_check_compile_flag.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_CHECK_MIC
#
# DESCRIPTION
#
#   Checks that selected compilers are able to build binaries for Intel K1OM
#   architecture.
#   Issues a warning when LD_LIBRARY_PATH variable is set, as it can cause
#   problems later in the link stage.
#   Sets variables host_dep_CXXFLAGS and host_dep_LDFLAGS.
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

AC_DEFUN([AX_CHECK_MIC],
[

AC_LANG_PUSH([C++])

if x$CXX != xicpc; then
  supported_compiler=no
else
  AX_CHECK_COMPILE_FLAG([-mmic],
    [supported_compiler=yes],
    [supported_compiler=no],
    [-Werror])
fi

AS_IF([test $supported_compiler = no],[
    AC_MSG_ERROR([
-------------------------------
Compilation for Intel MIC detected, but selected compiler
does not support it.
Please, specify CC=icc and CXX=icpc when cross-compiling for MIC architectures.
Also make sure that they support -mmic compiler and linker flag.
-------------------------------])
])

if test x$LD_LIBRARY_PATH != x; then
  AC_MSG_WARN([
-------------------------------
Cross-compiling for Intel MIC architecture.
The use of LD_LIBRARY_PATH is discouraged for this build, as
binary incompatible libraries may interfer in the link process.
-------------------------------])
fi

CXXFLAGS+=-mmic
LDFLAGS+=-mmic

AC_LANG_POP([C++])

])dnl AX_CHECK_MIC

