AC_DEFUN([AX_CHECK_MEMKIND],[
dnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnl
dnl                          jemalloc / memkind                             dnl
dnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnl
LDFLAGS_BKP="$LDFLAGS"
LIBS_BKP="$LIBS"

AC_ARG_WITH([jemalloc],
   AS_HELP_STRING([--with-jemalloc=dir], [Directory of jemalloc installation]),
   [
      JEMALLOC_LDFLAGS="-L$withval/lib -Wl,-rpath,$withval/lib"
   ]
)
LDFLAGS="$LDFLAGS $JEMALLOC_LDFLAGS"
AC_SEARCH_LIBS([je_malloc], [jemalloc], [JEMALLOC_LIBS="-ljemalloc"], [JEMALLOC_LIBS=""], [])

if test "JEMALLOC_LIBS"x != x ;
then
   MEMKIND_LDFLAGS="$JEMALLOC_LDFLAGS"
   AC_ARG_WITH([memkind],
      AS_HELP_STRING([--with-memkind=dir], [Directory of memkind installation]),
      [
         MEMKIND_CFLAGS="-I$withval/include"
         MEMKIND_LDFLAGS="$MEMKIND_LDFLAGS -L$withval/lib"
      ]
   )

   LDFLAGS="$LDFLAGS $MEMKIND_LDFLAGS"
   AC_SEARCH_LIBS([memkind_malloc], [memkind], [MEMKIND_LIBS="-lmemkind"], [MEMKIND_LIBS=""], [-ljemalloc])
fi

LDFLAGS="$LDFLAGS_BKP"
LIBS="$LIBS_BKP"
AM_CONDITIONAL([MEMKIND_SUPPORT], test "$MEMKIND_LIBS"x != x )
AC_SUBST([JEMALLOC_LIBS])
AC_SUBST([MEMKIND_LIBS])
AC_SUBST([MEMKIND_LDFLAGS])
AC_SUBST([MEMKIND_CFLAGS])
dnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnldnl

])
