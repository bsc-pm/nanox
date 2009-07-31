
AC_INIT([nanox], [0.01], [alex.duran@bsc.es])

# Environment
AC_CANONICAL_BUILD
AC_CANONICAL_SYSTEM
AC_CANONICAL_TARGET

# Automake initialization
AM_INIT_AUTOMAKE([-Wall -Werror dist-bzip2])

# Check for programs
AC_PROG_LIBTOOL
AC_PROG_CC
AC_PROG_CPP
AC_PROG_CXX
AM_PROG_AS

# Check for libraries

# Check for header files
AC_HEADER_STDC

# Check for typedefs, structures, and compiler characteristics
AC_C_CONST

# Check for library functions
NANOS_CONFIG_LIBS=
AC_SUBST([NANOS_CONFIG_LIBS])

# Check for architecture
ARCHITECTURES=
OS=

AC_SUBST([ARCHITECTURES])
AC_SUBST([OS])
AC_SUBST([SMP_ARCH])

AC_MSG_CHECKING([for architecture])
case $host in
  i?86-*-linux-gnu)
    OS=unix-os
    ARCHITECTURES+=smp
    SMP_ARCH=x86
    ;;
  x86_64-*-linux-gnu)
    OS=unix-os
    ARCHITECTURES+=smp
    SMP_ARCH=x86-64
    ;;
  ia64-*-linux)
    OS=unix-os
    ARCHITECTURES+=smp
    SMP_ARCH=ia64
    ;;
  *)
    AC_MSG_ERROR([unsupported architecture $host])
  ;;
esac

# Add architecture libraries
for arch in $ARCHITECTURES; do
	NANOS_CONFIG_LIBS+=../arch/$arch/lib$arch.la
done

AC_MSG_RESULT([$ARCHITECTURES/$OS])

# Enable performance version
performance_enabled="no"
AC_MSG_CHECKING([if performance version was requested])
AC_ARG_ENABLE([performance],
  AS_HELP_STRING([--enable-performance], [Enables generation of performance version]),
  [
    if test x$enableval = xyes -o x$enableval = x;
    then
        performance_enabled="yes"
        AC_MSG_RESULT([yes])
    else if test x$enableval = xno;
         then
            AC_MSG_RESULT([no])
            performance_enabled="no"
         else
            AC_MSG_ERROR([This option can only be given 'yes' or 'no' values])
         fi
    fi
  ],
  [
      performance_enabled="no"
      AC_MSG_RESULT([yes])
  ]
)

AM_CONDITIONAL([PERFORMANCE], [test "$performance_enabled" = yes])

# Output files
AC_CONFIG_HEADERS([config.h])

AC_CONFIG_FILES([Makefile
                 src/Makefile
                 src/core/Makefile
		 src/support/Makefile
		 src/sched/Makefile
		 src/apis/Makefile
		 src/apis/c/Makefile
                 src/arch/Makefile
		 src/arch/unix-os/Makefile
		 src/arch/smp/Makefile
		 src/arch/fpga/Makefile
		 src/arch/cell/Makefile
		 src/arch/gpu/Makefile
		 test/Makefile
		 doc/Makefile
                 ])

AC_OUTPUT

echo ""
echo "Configuration summary of Nanos++"
echo "================================"
echo ""
echo "Configured architectures:" $ARCHITECTURES
echo ""
