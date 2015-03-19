# ===========================================================================
#   http://www.gnu.org/software/autoconf-archive/ax_check_compile_flag.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_CHECK_MIC(FLAG, [ACTION-SUCCESS], [ACTION-FAILURE], [EXTRA-FLAGS], [INPUT])
#
# DESCRIPTION
#
#   Check whether OpenCL path to the headers and libraries are correctly specified.
#
#   ACTION-SUCCESS/ACTION-FAILURE are shell commands to execute on
#   success/failure.
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

AC_DEFUN([AX_CHECK_OPENCL],
[
# AC_PREREQ(2.59)dnl for _AC_LANG_PREFIX
# AS_VAR_PUSHDEF([CACHEVAR],[ax_cv_check_[]_AC_LANG_ABBREV[]flags_$4_$1])dnl
# AC_CACHE_CHECK([whether _AC_LANG compiler accepts $1], CACHEVAR, [
#   ax_check_save_flags=$[]_AC_LANG_PREFIX[]FLAGS
#   _AC_LANG_PREFIX[]FLAGS="$[]_AC_LANG_PREFIX[]FLAGS $4 $1"
#   AC_COMPILE_IFELSE([m4_default([$5],[AC_LANG_PROGRAM()])],
#     [AS_VAR_SET(CACHEVAR,[yes])],
#     [AS_VAR_SET(CACHEVAR,[no])])
#   _AC_LANG_PREFIX[]FLAGS=$ax_check_save_flags])
# AS_IF([test x"AS_VAR_GET(CACHEVAR)" = xyes],
#   [m4_default([$2], :)],
#   [m4_default([$3], :)])
# AS_VAR_POPDEF([CACHEVAR])dnl
AC_PREREQ(2.59)dnl for _AC_LANG_PREFIX

if x$CXX != xicpc; then
    AC_MSG_ERROR([
-------------------------------
Compilation for Intel MIC detected, but selected compiler
does not support it.
Please, specify C=icc and CXX=icpc when cross-compiling for MIC architectures.
Also make sure that they support -mmic compiler and linker flag.
-------------------------------])
fi

AX_CHECK_COMPILE_FLAG([-mmic],
  [],[
    AC_MSG_ERROR([
-------------------------------
Compilation for Intel MIC detected, but selected compiler
does not support it.
Please, specify CC=icc and CXX=icpc when cross-compiling for MIC architectures.
Also make sure that they support -mmic compiler and linker flag.
-------------------------------])
  ],[-Werror])

if test x$LD_LIBRARY_PATH != x; then
  AC_MSG_WARN([
-------------------------------
Cross-compiling for Intel MIC architecture.
The use of LD_LIBRARY_PATH is discouraged for this build, as
binary incompatible libraries may interfer in the link process.
-------------------------------])
fi

AC_SUBST([INTEL_MIC_FLAGS],[-mmic])

])dnl AX_CHECK_MIC
