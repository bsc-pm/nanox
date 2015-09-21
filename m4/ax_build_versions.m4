
AC_DEFUN([AX_BUILD_VERSIONS],[

VERSIONS=

# Preprocessor, compiler flags and default availability
# Performance
m4_define(performance_default,yes)
performance_CPPFLAGS=
performance_CXXFLAGS="-O3"

# Debug
m4_define(debug_default,yes)
debug_CPPFLAGS="-DNANOS_DEBUG_ENABLED"
debug_CXXFLAGS="-O0 -g2 $no_inline_flag"

# Instrumentation
m4_define(instrumentation_default,yes)
instrumentation_CPPFLAGS="-DNANOS_INSTRUMENTATION_ENABLED"
instrumentation_CXXFLAGS="-O3"

# Instrumentation-debug
m4_define(instrumentation_debug_default,yes)
instrumentation_debug_CPPFLAGS="-DNANOS_DEBUG_ENABLED -DNANOS_INSTRUMENTATION_ENABLED"
instrumentation_debug_CXXFLAGS="-O0 -g2 $no_inline_flag"

m4_foreach([version_name],[[performance], [debug], [instrumentation], [instrumentation-debug]],[
  _ax_enable_version(version_name)
])

if test "x$VERSIONS" = x; then
  AC_MSG_ERROR([
------------------------------
At least one version needs to be compiled
------------------------------])
fi
AC_SUBST([VERSIONS])

])dnl AX_BUILD_VERSIONS

# Helper function
# Includes enable/disable configuration argument to 'help' dialog
# Checks default values and explicit user configuration
# Substitutes autoconf and automake variables
AC_DEFUN([_ax_enable_version],[
  AS_VAR_PUSHDEF([version_default],[$1_default])
  AS_VAR_PUSHDEF([version_enabled],[$1_enabled])
  AS_VAR_PUSHDEF([version_dir],[$1_dir])
  AS_VAR_PUSHDEF([cppflags],[$1_CPPFLAGS])
  AS_VAR_PUSHDEF([cxxflags],[$1_CXXFLAGS])

  AC_MSG_CHECKING([if version_name version is enabled])
  AC_ARG_ENABLE($1,
    [m4_if(version_default,yes,
       AS_HELP_STRING(--enable-$1,Enable generation of $1 version),
       AS_HELP_STRING(--disable-$1,Disable generation of $1 version)
    )],
    [version_enabled=${enableval}],
    [version_enabled=version_default])
  
  AC_MSG_RESULT([$version_enabled])
  
  if test x$version_enabled = xyes; then
    VERSIONS+="$1 "
  fi
  
  version_dir='${libdir}/$1'
  AM_CONDITIONAL(version_enabled, [test x$version_enabled = xyes])
  AC_SUBST(version_dir)
  AC_SUBST(cppflags)
  AC_SUBST(cxxflags)

  AS_VAR_POPDEF([version_default])
  AS_VAR_POPDEF([version_enabled])
  AS_VAR_POPDEF([version_dir])
  AS_VAR_PUSHDEF([cppflags])
  AS_VAR_PUSHDEF([cxxflags])
])dnl check_version

