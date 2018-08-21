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
#   AX_CHECK_SQLITE3
#
# DESCRIPTION
#
#   Check whether a valid SQLite3 library is available, and the path to the headers 
#   and libraries are correctly specified.
#

AC_DEFUN([AX_CHECK_SQLITE3],
[
#Check if SQLITE3 lib is installed.
AC_ARG_WITH(sqlite3,
  [AS_HELP_STRING([--with-sqlite3,--with-sqlite3=PATH],
                [search in system directories or specify prefix directory for installed SQLITE3 package.])],
  [
    # Check if the user provided a valid PATH
    AS_IF([test -d "$withval"],[
      sqlite3=yes
      sqlite3_path_provided=yes
    ],[
      sqlite3=$withval
      sqlite3_path_provided=no
    ])dnl
  ],[
    # Default: check if sqlite3 is available
    sqlite3=yes
    sqlite3_path_provided=no
  ])

# If the user specifies --with-sqlite3, $with_sqlite3 value will be 'yes'
#                       --without-sqlite3, $with_sqlite3 value will be 'no'
#                       --with-sqlite3=somevalue, $with_sqlite3 value will be 'somevalue'
AS_IF([test "$sqlite3_path_provided" = yes],[
  sqlite3inc="-I$with_sqlite3/include"
  sqlite3lib="-I$with_sqlite3/lib"
])dnl

AS_IF([test "$sqlite3" = yes],[
  AX_VAR_PUSHVALUE([CPPFLAGS],[$CPPFLAGS $sqlite3inc])
  AX_VAR_PUSHVALUE([CXXFLAGS])
  AX_VAR_PUSHVALUE([LDFLAGS],[$LDFLAGS $sqlite3lib])
  AX_VAR_PUSHVALUE([LIBS],[])

  # Check for header
  AC_CHECK_HEADERS([sqlite3.h],
    [sqlite3=yes],
    [sqlite3=no])

  # Look for sqlite3_init function in libsqlite3
  AS_IF([test "$sqlite3" = yes],[
      AC_SEARCH_LIBS([sqlite3_libversion_number],
                [sqlite3],
                [sqlite3=yes],
                [sqlite3=no])
  ])dnl
  
  sqlite3libs="$LIBS"

  AX_VAR_POPVALUE([CPPFLAGS])
  AX_VAR_POPVALUE([CXXFLAGS])
  AX_VAR_POPVALUE([LDFLAGS])
  AX_VAR_POPVALUE([LIBS])
])dnl

AC_SUBST([sqlite3inc])
AC_SUBST([sqlite3lib])
AC_SUBST([sqlite3libs])
AM_CONDITIONAL([SQLITE3_SUPPORT],[test "$sqlite3" = yes])

])dnl AX_CHECK_SQLITE3

