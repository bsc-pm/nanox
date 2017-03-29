#
# SYNOPSIS
#
#   AX_CHECK_DLB
#
# DESCRIPTION
#
#   Check Dynamic Load Balancing (DLB) support
#

AC_DEFUN([AX_CHECK_DLB],
[
   AC_MSG_CHECKING([for DLB])
   AC_ARG_WITH([dlb],
      [AS_HELP_STRING([--with-dlb@<:@=DIR@:>@], [build with DLB library support])],
      [], dnl Implicit: with_dlb=$withvalue
      [with_dlb=no]
   )
   AC_MSG_RESULT([$with_dlb])

   AS_IF([test "x$with_dlb" != xno], [
      AS_IF([test -d "$with_dlb"], [
         AS_IF([test -d "$with_dlb/include"], [dlbcppflags="-I$with_dlb/include"])
         AS_IF([test -d "$with_dlb/lib"], [dlbldflags="-L$with_dlb/lib"])
      ])

      ### DLB INCLUDES
      AC_LANG_PUSH([C++])
      AX_VAR_PUSHVALUE([CPPFLAGS], [$dlbcppflags])
      AC_CHECK_HEADERS([dlb.h], [
         # header found, do nothing
      ], [
         AS_IF([test "x$with_dlb" != xcheck], [AC_MSG_ERROR([Cannot find DLB headers])])
         with_dlb=no
      ])
      AX_VAR_POPVALUE([CPPFLAGS])
      AC_LANG_POP([C++])
   ])

   AS_IF([test "x$with_dlb" != xno], [
      ### DLB LIBS
      AC_LANG_PUSH([C++])
      AX_VAR_PUSHVALUE([LIBS], [""])
      AX_VAR_PUSHVALUE([LDFLAGS], [$dlbldflags])
      AC_SEARCH_LIBS([DLB_Init], [dlb], [
         dlblibs="$LIBS"
      ], [
         AS_IF([test "x$with_dlb" != xcheck], [AC_MSG_ERROR([Cannot find DLB libraries])])
         with_dlb=no
      ])
      AX_VAR_POPVALUE([LDFLAGS])
      AX_VAR_POPVALUE([LIBS])
      AC_LANG_POP([C++])
   ])

   AS_IF([test "x$with_dlb" != xno], [
      AC_DEFINE([DLB],[],[Enables DLB support.])
      AC_SUBST([HAVE_DLB], [DLB])
   ], [
      AC_SUBST([HAVE_DLB], [NO_DLB])
   ])
   AC_SUBST([dlbcppflags])
   AC_SUBST([dlbldflags])
   AC_SUBST([dlblibs])
])
