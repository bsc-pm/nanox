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
      # Header extrae_version.h exists until Extrae 3.4
      # API Extrae_get_version exists since Extrae 3.4

      AC_MSG_CHECKING([for Extrae version])

      extrae_version_h=$with_extrae/include/extrae_version.h
      AS_IF([test -e $extrae_version_h], [
         extrae_version_string=$(sed -ne '/^\s*#\s*define\s\+EXTRAE_VERSION .*$/p' $extrae_version_h)
         extrae_version=$(echo $extrae_version_string | sed -e 's/#define EXTRAE_VERSION EXTRAE_VERSION_NUMBER(\([0-9]*\),\([0-9]*\),\([0-9]*\).*$/\1\2\3/')
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
                  printf("%1d%1d%1d\n", major, minor, revision);
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

      AS_IF([test "$extrae_version" -ge 240],[
            MPITRACE_HOME="$withval"
            MPITRACE_INC="$withval/include"
            MPITRACE_LIB="$withval/lib"
            AS_IF([test -d "$MPITRACE_HOME/lib64"],[
               MPITRACE_LIB="$MPITRACE_HOME/lib64"
            ])
            MPITRACE_BIN="$withval/bin"
            AC_MSG_RESULT([checking if Extrae library is compatible... yes])
      ],[
         AC_MSG_ERROR([checking if Extrae library is compatible... no (Extrae >= 2.4 needed)])
      ])
   ])

   AC_SUBST([MPITRACE_HOME])
   AC_SUBST([MPITRACE_INC])
   AC_SUBST([MPITRACE_LIB])
   AC_SUBST([MPITRACE_BIN])

   AM_CONDITIONAL([instrumentation_EXTRAE], test x"$MPITRACE_HOME" != x)
])
