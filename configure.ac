
AC_INIT([nanox], [0.5], [alex.duran@bsc.es])

# Environment
AC_CANONICAL_BUILD
AC_CANONICAL_SYSTEM
AC_CANONICAL_TARGET

# Automake initialization
AM_INIT_AUTOMAKE([-Wall -Werror dist-bzip2])
AC_CONFIG_MACRO_DIR([m4])

# Check for programs
AC_PROG_LIBTOOL
AC_PROG_CC
AC_PROG_CPP
AM_PROG_CC_C_O
AC_PROG_CXX
AM_PROG_AS

# Check for libraries

# Check for header files
AC_HEADER_STDC

# Check for typedefs, structures, and compiler characteristics
AC_C_CONST

# Check for library functions
NANOS_CONFIG_LIBDIRS=
NANOS_CONFIG_LIBS=
AC_SUBST([NANOS_CONFIG_LIBDIRS])
AC_SUBST([NANOS_CONFIG_LIBS])

# Check for architecture
ARCHITECTURES=
OS=

AC_SUBST([ARCHITECTURES])
AC_SUBST([OS])
AC_SUBST([SMP_ARCH])

AC_MSG_CHECKING([for architecture])
smp_ult=yes
case $host in
  i?86-*-linux*)
    OS=unix-os
    ARCHITECTURES+=smp
    SMP_ARCH=x86
    ;;
  x86_64-*-linux*)
    OS=unix-os
    ARCHITECTURES+=smp
    SMP_ARCH=x86-64
    ;;
  ia64-*-linux*)
    OS=unix-os
    ARCHITECTURES+=smp
    SMP_ARCH=ia64
    ;;
  powerpc-*-linux* | powerpc64-*-linux*)
    OS=unix-os
    ARCHITECTURES+=smp
    SMP_ARCH=ppc32
    CXXFLAGS+=" -m32"
    CFLAGS+=" -m32"

    # Check if it is a Cell system
    if cat /proc/cpuinfo | grep Cell &> /dev/null ; then
      spu_valid=yes
      AC_ARG_VAR([SPUCC],[SPU C compiler])
      AC_ARG_VAR([PPUEMBED],[SPU to PPU embedding tool])
      AC_CHECK_PROG( [SPUCC], [spu-cc], spu-cc, no)
      if test x$SPUCC = xno ; then
	AC_MSG_WARN([spu-cc not found. Disabling SPU support]) 
        spu_valid=no
      else 
        AC_CHECK_PROG( [PPUEMBED], [ppu-embedspu], ppu-embedspu, no)
        if test x$PPUEMBED = xno ; then
          AC_MSG_WARN([ppu-embedspu not found. Disabling SPU support]) 
          spu_valid=no
        else
           AC_ARG_WITH([cellsdk],
              AS_HELP_STRING([--with-cellsdk=dir], [Directory of Cell SDK installation]),
              [
		CELLSDK="${withval}"
              ],
              [
		CELLSDK=/opt/cell/sdk/usr
              ])
           CELLSDK_INC="$CELLSDK/include"
           CELLSDK_LIB="$CELLSDK/lib"

	   AC_SUBST([CELLSDK_INC])
	   AC_SUBST([CELLSDK_LIB])

	   NANOS_CONFIG_LIBDIRS+=" -L$CELLSDK_LIB"
	   NANOS_CONFIG_LIBS+=" -lspe2"
        fi
      fi
      if test x$spu_valid = xyes; then
        ARCHITECTURES+=" spu"
      fi
    fi

    ;;
  *)
    ARCHITECTURES+=smp
    SMP_ARCH=unknown
    smp_ult=no
    ;;
esac

# Add architecture libraries
for arch in $ARCHITECTURES; do
	NANOS_CONFIG_LIBS+=" ../arch/$arch/lib$arch.la"
done

AM_CONDITIONAL([SMP_SUPPORTS_ULT],[test "$smp_ult"=yes])

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

# Check for atomic builtins in GCC
AC_CHECK_GXX_ATOMIC_BUILTINS()

AM_CONDITIONAL([PERFORMANCE], [test "$performance_enabled" = yes])

AC_ARG_WITH([mpitrace],
           AS_HELP_STRING([--with-mpitrace=dir], [Directory of mpitrace installation]),
           [
                MPITRACE_HOME="$withval"
                MPITRACE_INC="$withval/include"
                MPITRACE_LIB="$withval/lib"
                MPITRACE_BIN="$withval/bin"
           ],
           [
                MPITRACE_HOME=""
                MPITRACE_INC=""
                MPITRACE_LIB=""
                MPITRACE_BIN=""
           ]
)
	   AC_SUBST([MPITRACE_HOME])
	   AC_SUBST([MPITRACE_INC])
	   AC_SUBST([MPITRACE_LIB])
	   AC_SUBST([MPITRACE_BIN])


AM_CONDITIONAL([INSTRUMENTOR_PARAVER], test x"$MPITRACE_HOME" != x)

# Output files
AC_CONFIG_HEADERS([config.h])

AC_CONFIG_FILES([Makefile
                 src/Makefile
                 src/core/Makefile
		 src/support/Makefile
		 src/plugins/Makefile
		 src/plugins/sched/Makefile
		 src/plugins/throttle/Makefile
		 src/plugins/barr/Makefile
		 src/plugins/slicers/Makefile
		 src/plugins/instrumentors/Makefile
		 src/apis/Makefile
		 src/apis/c/Makefile
		 src/pms/Makefile
		 src/pms/openmp/Makefile
		 src/pms/ss/Makefile
		 src/pms/chapel/Makefile
                 src/arch/Makefile
		 src/arch/unix-os/Makefile
		 src/arch/smp/Makefile
		 src/arch/fpga/Makefile
		 src/arch/spu/Makefile
		 src/arch/gpu/Makefile
		 src/utils/Makefile
		 test/Makefile
		 test/gens/Makefile
		 doc/Makefile
                 doc/doxy-nanox
                 ])

AC_OUTPUT

echo ""
echo "Configuration summary of Nanos++"
echo "================================"
echo ""
echo "Configured architectures:" $ARCHITECTURES
echo ""
