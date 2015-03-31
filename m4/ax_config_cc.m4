#
# SYNOPSIS
#
#   AX_CONFIG_CC
#
# DESCRIPTION
#
#   Sets default flags depending on which C++ compiler is being used.
#   
# LICENSE
#
#   Copyright (c) 2008 Guido U. Draheim <guidod@gmx.de>
#   Copyright (c) 2011 Maarten Bosmans <mkbosmans@gmail.com>
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

AC_DEFUN([AX_CONFIG_CC],[
AC_PREREQ(2.59)

cc_dep_CPPFLAGS=
cc_dep_CXXFLAGS=
cc_dep_LDFLAGS=

if test x$ac_cv_cxx_compiler_gnu = xyes; then
  cc_dep_CPPFLAGS="-include new_decl.hpp -include config.h"
  cc_dep_CXXFLAGS="$cc_dep_CXXFLAGS -Wall -Wextra -Werror -Wshadow -Wmissing-declarations -Wno-unused-parameter"
  no_inline_flag=-fno-inline
fi

if [[[ x$CXX =~ .*"xlC" ]]]; then
     cc_dep_CPPFLAGS="-qinclude=\"new_decl.hpp\" -qinclude=\"config.h\""
	  cc_dep_CXXFLAGS="$cc_dep_CXXFLAGS -qlanglvl=variadictemplates"
	  cc_dep_LDFLAGS="$cc_dep_LDFLAGS -Wl,-z,muldefs"
	  no_inline_flag=-qno-inline
fi

# internal headers are put in a separate 
AC_SUBST([cc_dep_CPPFLAGS])
AC_SUBST([cc_dep_CXXFLAGS])
AC_SUBST([cc_dep_LDFLAGS])

]) # AX_CONFIG_CC
