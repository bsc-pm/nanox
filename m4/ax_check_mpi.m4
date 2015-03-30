# ===========================================================================
#   http://www.gnu.org/software/autoconf-archive/ax_check_compile_flag.html
# ===========================================================================
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

AC_DEFUN([AX_CHECK_MPI],
[
AC_MSG_CHECKING([if MPI Offload architecture was requested])
AC_ARG_ENABLE([mpi], AS_HELP_STRING([--enable-mpi], [Enables mpi offload support/architecture]),
				 [enable_mpi=$enableval],[enable_mpi=yes])
AC_MSG_RESULT([$enable_mpi])

if test "x$enable_mpi" = xyes; then
	ARCHITECTURES="$ARCHITECTURES mpi"
	AC_DEFINE([MPI_DEV],[],[Specifies whether MPI Offload architecture is enabled or not.])
fi

])dnl AX_CHECK_MPI

