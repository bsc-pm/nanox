#
# SYNOPSIS
#
#   AX_CHECK_CLUSTER
#
# DESCRIPTION
#
# Provides a configurable enable/disable argument, so that the user can
# explicitly require this architecture support.
# Requires GASNet
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
AC_REQUIRE([AX_CHECK_GASNET])
AC_PREREQ(2.59)

# Check if user specifically requested Cluster architecture
AC_MSG_CHECKING([if Cluster architecture was requested])
AC_ARG_ENABLE([cluster], AS_HELP_STRING([--enable-cluster], [Enables cluster architecture]),
    [cluster_enabled=${enableval}],
    [cluster_enabled=no])

AC_MSG_RESULT([$cluster_enabled])

AS_IF([ test x$cluster_enabled = xyes ],[
  AS_IF([ test "x$gasnet_available_conduits" = x],
      [AC_MSG_ERROR([
------------------------------
Cluster architecture requires GASNet library.
------------------------------])
  ])

  ARCHITECTURES="$ARCHITECTURES cluster"
  AC_DEFINE([CLUSTER_DEV],[],[Indicates the presence of the Cluster arch plugin.])
])

]) dnl AX_CHECK_CLUSTER
