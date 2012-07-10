AC_DEFUN([AC_CHECK_GXX_ATOMIC_BUILTINS], [
   AC_MSG_CHECKING([for atomic builtins in GCC])
   AC_LANG_PUSH([C++])
   AC_LINK_IFELSE(
      [AC_LANG_PROGRAM([],
         [[
          int foo = -10;
          int bar = 10;
          __sync_fetch_and_add(&foo, bar);
         ]]
      )],
      [ac_cv_have_builtins=yes],
      [ac_cv_have_builtins=no])
      if test x"$ac_cv_have_builtins" = xno; then
         CXXFLAGS+=" -march=i686"
         AC_LINK_IFELSE(
            [AC_LANG_PROGRAM([],
               [[
                int foo = -10;
                int bar = 10;
                __sync_fetch_and_add(&foo, bar);
               ]]
            )],
            [ac_cv_have_builtins_insist=yes],
            [ac_cv_have_builtins_insist=no])
         if test x"$ac_cv_have_builtins_insist" = xno; then
            AC_MSG_RESULT([no])
            AC_MSG_ERROR([Gcc atomic builtins are necessary to compile this library])
         else
            AC_MSG_RESULT([yes])
         fi
      else
         AC_MSG_RESULT([yes])
      fi
# __sync_bool_compare_and_swap_8
   AC_MSG_CHECKING([if __sync_bool_compare_and_swap_8 is available])
   AC_LINK_IFELSE(
            [AC_LANG_PROGRAM([],
               [[
                    typedef unsigned long long uint64;
                    uint64 i;
                    __sync_bool_compare_and_swap (&i, 0, 1);
               ]]
            )],
            [available_builtin=yes],
            [available_builtin=no])
   if test "$available_builtin" = "yes"; then
       AC_MSG_RESULT([yes])
       AC_DEFINE(HAVE_SYNC_BOOL_COMPARE_AND_SWAP_8, 1,
       [Define to 1 if the compiler provides the __sync_bool_compare_and_swap function for uint64])
   else
       AC_MSG_RESULT([no])
   fi

# __sync_add_and_fetch_8
   AC_MSG_CHECKING([if __sync_add_and_fetch_8 is available])
   AC_LINK_IFELSE(
      [AC_LANG_PROGRAM([],
               [[
                  typedef unsigned long long uint64;
                  uint64 i;
                  __sync_add_and_fetch (&i, 1);
               ]]
            )],
      [available_builtins=yes], [available_builtins=no])
   if test x$available_builtins = xyes; then
      AC_MSG_RESULT([yes])
      AC_DEFINE(HAVE_SYNC_ADD_AND_FETCH_8, 1, [Define to 1 if the compiler provides the __sync_add_and_fetch function for uint64])
   else
      AC_MSG_RESULT([no])
   fi
# __sync_sub_and_fetch_8
    AC_MSG_CHECKING([if __sync_sub_and_fetch_8 is available])
    AC_LINK_IFELSE(
       [AC_LANG_PROGRAM([],
          [[
             typedef unsigned long long uint64;
             uint64 i;
             __sync_sub_and_fetch (&i, 1);
          ]]
          )],
             [available_builtins=yes],
             [available_builtins=no])
    if test "$available_builtins" = "yes"; then
        AC_MSG_RESULT([yes])
        AC_DEFINE(HAVE_SYNC_SUB_AND_FETCH_8, 1,
        [Define to 1 if the compiler provides the __sync_sub_and_fetch function for uint64])
    else
        AC_MSG_RESULT([no])
    fi
# __sync_or_and_fetch_8
    AC_MSG_CHECKING([if __sync_or_and_fetch_8 is available])
    AC_LINK_IFELSE(
       [AC_LANG_PROGRAM([],
          [[
             typedef unsigned long long uint64;
             uint64 i;
             __sync_or_and_fetch (&i, 1);
          ]]
          )],
             [available_builtins=yes],
             [available_builtins=no])
    if test "$available_builtins" = "yes"; then
        AC_MSG_RESULT([yes])
        AC_DEFINE(HAVE_SYNC_OR_AND_FETCH_8, 1,
        [Define to 1 if the compiler provides the __sync_or_and_fetch function for uint64])
    else
        AC_MSG_RESULT([no])
    fi
# __sync_and_and_fetch_8
    AC_MSG_CHECKING([if __sync_and_and_fetch_8 is available])
    AC_LINK_IFELSE(
       [AC_LANG_PROGRAM([],
          [[
             typedef unsigned long long uint64;
             uint64 i;
             __sync_and_and_fetch (&i, 1);
          ]]
          )],
             [available_builtins=yes],
             [available_builtins=no])
    if test "$available_builtins" = "yes"; then
        AC_MSG_RESULT([yes])
        AC_DEFINE(HAVE_SYNC_AND_AND_FETCH_8, 1,
        [Define to 1 if the compiler provides the __sync_and_and_fetch function for uint64])
    else
        AC_MSG_RESULT([no])
    fi
# __sync_xor_and_fetch_8
    AC_MSG_CHECKING([if __sync_xor_and_fetch_8 is available])
    AC_LINK_IFELSE(
       [AC_LANG_PROGRAM([],
          [[
             typedef unsigned long long uint64;
             uint64 i;
             __sync_xor_and_fetch (&i, 1);
          ]]
          )],
             [available_builtins=yes],
             [available_builtins=no])
    if test "$available_builtins" = "yes"; then
        AC_MSG_RESULT([yes])
        AC_DEFINE(HAVE_SYNC_XOR_AND_FETCH_8, 1,
        [Define to 1 if the compiler provides the __sync_xor_and_fetch function for uint64])
    else
        AC_MSG_RESULT([no])
    fi
   AC_LANG_POP([C++])
])
