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
AC_BEFORE([AC_PROG_CXX].[$0])

AC_ARG_VAR([LD],[Linker command])

cc_dep_CPPFLAGS=

AC_LANG_PUSH([C])
AX_COMPILER_VENDOR
AC_LANG_POP([C])

AC_LANG_PUSH([C++])
AX_COMPILER_VENDOR
AC_LANG_POP([C++])

# Both  C and C++ compiler  vendors must be the same
AS_IF([test "$ax_cv_c_compiler_vendor" != "$ax_cv_cxx_compiler_vendor"],
  [AC_MSG_ERROR([
-------------------------------
C and C++ compiler vendors differ. Please,
make sure both compiler vendors are the same.
C compiler:   $ax_cv_c_compiler_vendor
C++ compiler: $ax_cv_cxx_compiler_vendor
-------------------------------])
  ])

AS_CASE([$ax_cv_cxx_compiler_vendor],
 [ibm],
   [
     cc_dep_CPPFLAGS="-qinclude=\"config.h\" -qinclude=\"nanox-config.h\""
     cc_dep_CXXFLAGS="-qinclude=\"new_decl.hpp\""
     AX_APPEND_FLAG([-qlanglvl=variadictemplates],[cc_dep_CXXFLAGS])
     AX_APPEND_FLAG([-Wl,-z,muldefs],[LDFLAGS])
     no_inline_flag=-qno-inline
   ],
 [
   # Default: use -include flag
   cc_dep_CPPFLAGS="-include \"config.h\" -include \"nanox-config.h\""
   cc_dep_CXXFLAGS="-include \"new_decl.hpp\""
   AX_APPEND_FLAG([-Wall -Wextra -Wshadow -Wmissing-declarations -Wno-unused-parameter -Wno-missing-field-initializers -Werror],[cc_dep_CXXFLAGS])
   no_inline_flag=-fno-inline
 ])

AS_IF([test "$ax_cv_cxx_compiler_vendor" = "gnu"],[
   AC_CACHE_CHECK([gcc version],[ax_cv_gcc_version],[
      ax_cv_gcc_version="`$CC -dumpversion`"
      # GCC 6.0 defaults to -std=c++11
      AX_COMPARE_VERSION([$ax_cv_gcc_version], [ge], [6], [
         cc_dep_CXXFLAGS+=" -std=c++98"
      ])
   ])
])

AC_SUBST([cc_dep_CPPFLAGS])
AC_SUBST([cc_dep_CXXFLAGS])
AC_SUBST([no_inline_flag])

]) dnl AX_CONFIG_CC
