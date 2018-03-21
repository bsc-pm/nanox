#
# SYNOPSIS
#
#   AX_CHECK_EXTRAE
#
# DESCRIPTION
#
#   Check Extrae support
#

AC_DEFUN([AX_CHECK_EXTRAE],
[
   AC_REQUIRE([AC_PROG_AWK])

   MPITRACE_HOME=""
   MPITRACE_INC=""
   MPITRACE_LIB=""
   MPITRACE_BIN=""

   AC_MSG_CHECKING([for Extrae])
   AC_ARG_WITH([extrae],
      AS_HELP_STRING([--with-extrae=PATH], [build with Extrae support]),
      [], dnl Implicit: with_extrae=$withvalue
      [with_extrae=no]
   )
   AC_MSG_RESULT([$with_extrae])

   AS_IF([test "x$with_extrae" != xno], [
      # Extrae 2.4 or higher is needed
      # Header extrae_version.h exists until Extrae 3.4 and again since 3.5.3
      # API Extrae_get_version exists since Extrae 3.4

      AC_MSG_CHECKING([for Extrae version])

      extrae_version_h="$with_extrae/include/extrae_version.h"
      AS_IF([test -e "$extrae_version_h"], [
         # Obtain version through Extrae header
         extrae_version=$($AWK -v FS='[[(),]]' \
            '/EXTRAE_VERSION_NUMBER\([[0-9]],[[0-9]],[[0-9]]\)/{print $[2]"."$[3]"."$[4];}' \
            $extrae_version_h)
         AS_IF([test -z "$extrae_version"], [
            extrae_version=$($AWK \
               '
               BEGIN{major=""; minor=""; micro="";}
               /EXTRAE_MAJOR [[0-9]]/{major=$[3];}
               /EXTRAE_MINOR [[0-9]]/{minor=$[3];}
               /EXTRAE_MICRO [[0-9]]/{micro=$[3];}
               END{if (length(major)>0 && length(minor)>0 && length(micro)>0)
                     print major"."minor"."micro;}
               ' $extrae_version_h)
         ])
      ], [
         # Obtain version through Extrae API
         AC_LANG_PUSH([C])
         AC_LANG_CONFTEST([
            AC_LANG_SOURCE([[
               #include <stdio.h>
               void Extrae_get_version (unsigned *major, unsigned *minor, unsigned *revision);
               int main()
               {
                  unsigned major, minor, revision;
                  Extrae_get_version(&major, &minor, &revision);
                  printf("%1d.%1d.%1d\n", major, minor, revision);
                  return 0;
               }
            ]])
         ])
         AC_LANG_POP([C])

         AS_IF([$CC conftest.c -o conftest \
                  -lnanostrace -L$with_extrae/lib -Wl,-rpath,$with_extrae/lib \
                  2>&AS_MESSAGE_LOG_FD 1>&2], [
            extrae_version=$(./conftest)
         ])
         rm -f conftest
      ])

      AC_MSG_RESULT([$extrae_version])

      AS_IF([test -z "$extrae_version"],[
         AC_MSG_ERROR([could not find Extrae installation])
      ])

      AC_MSG_CHECKING([if Extrae library is compatible])
      AX_COMPARE_VERSION([$extrae_version],[ge],[2.4], [
         MPITRACE_HOME="$withval"
         MPITRACE_INC="$withval/include"
         MPITRACE_LIB="$withval/lib"
         AS_IF([test -d "$MPITRACE_HOME/lib64"],[
            MPITRACE_LIB="$MPITRACE_HOME/lib64"
         ])
         MPITRACE_BIN="$withval/bin"
         AC_MSG_RESULT([yes])
      ],[
         AC_MSG_ERROR([no (Extrae >= 2.4 needed)])
      ])
   ])

   AC_SUBST([MPITRACE_HOME])
   AC_SUBST([MPITRACE_INC])
   AC_SUBST([MPITRACE_LIB])
   AC_SUBST([MPITRACE_BIN])

   AM_CONDITIONAL([instrumentation_EXTRAE], test x"$MPITRACE_HOME" != x)
])
