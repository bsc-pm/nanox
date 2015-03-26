#
# SYNOPSIS
#
#   AX_CHECK_BGQ
#
# DESCRIPTION
#
#   Check that required header files for BlueGene/Q drivers are present.
#   Also finds whether -dynamic or -qnostaticbuild flags are necesary
#   to allow dynamic linkage.
#   Sets variables host_dep_CPPFLAGS and host_dep_LDFLAGS.
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

AC_DEFUN([AX_CHECK_BGQ],
[
AC_PREREQ(2.59)dnl for _AC_LANG_PREFIX

AC_LANG_PUSH([C])

bak_CPPFLAGS=$CPPFLAGS
bak_CFLAGS=$CFLAGS

# Old BG/Q cppflags: -I/bgsys/drivers/V1R2M1/ppc64 -I/bgsys/drivers/V1R2M1/ppc64/spi/include/kernel/cnk
# The latter one might not be needed as all the header files included inside Nanox belong to a different
# directory
# Includes used:
#ifdef IS_BGQ_MACHINE
#include <spi/include/kernel/location.h>
#include <spi/include/kernel/process.h>
#endif

CPPFLAGS=-I/bgsys/drivers/ppcfloor/ppc64
CFLAGS=

AC_CHECK_HEADER([spi/include/kernel/location.h],[location_h_present=yes])
AC_CHECK_HEADER([spi/include/kernel/process.h] ,[process_h_present=yes],[])

if "x$location_h_present" != xyes -o "x$process_h_present" != xyes; then
  AC_MSG_ERROR([
----------------------------
Could not find the following BlueGene/Q driver headers:
/bgsys/drivers/ppcfloor/ppc64/spi/include/kernel/location.h
/bgsys/drivers/ppcfloor/ppc64/spi/include/kernel/process.h
----------------------------])
fi

# Flags & Compiler dependent stuff
AX_CHECK_COMPILE_FLAG([-dynamic],
    [host_dep_LDFLAGS=-dynamic],
    [],[-Werror])
        
AX_CHECK_COMPILE_FLAG([-qnostaticlink],
    [host_dep_LDFLAGS=-qnostaticlink],
    [],[-Werror])

host_dep_CPPFLAGS=$CPPFLAGS

CPPFLAGS=bak_$CPPFLAGS
CFLAGS=bak_$CFLAGS

AC_LANG_POP([C])

if test "x$dyn_link_flag" = x; then
    AC_MSG_ERROR([
------------------------------
Specified compiler does not support
dynamic link in BlueGene/Q.
Failed to link using either -dynamic
or -qnostaticlink flags.
------------------------------])
fi

AC_DEFINE([IS_BGQ_MACHINE],[],[BlueGene/Q host compatibility is enabled.])

]) # AX_CHECK_BGQ
