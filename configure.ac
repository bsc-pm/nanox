AC_INIT([nanox], [0.14.1], [pm-tools@bsc.es])
AC_LANG([C++])
AC_CONFIG_SRCDIR([src/core/system.cpp])
AC_DEFINE_UNQUOTED([NANOX_CONFIGURE_ARGS], ["$0 $@"], [Configure line])

# Environment
AC_CANONICAL_BUILD
AC_CANONICAL_SYSTEM
AC_CANONICAL_TARGET

# Automake initialization
AM_INIT_AUTOMAKE([-Wall dist-bzip2 -Wno-portability subdir-objects foreign])
AC_CONFIG_MACRO_DIR([m4])

# Enable silent rules if available
supported_silent_rules=1
m4_ifdef([AM_SILENT_RULES],
         [AM_SILENT_RULES([yes])],
         [supported_silent_rules=0])

AM_CONDITIONAL([SUPPORTED_SILENT_RULES], test x$supported_silent_rules = x1)
 
# Nanos version check
# Make sure build_version is double-quoted!
AS_IF([test -e "$srcdir/.git"],[
   # Git directory found
   AC_PATH_PROG([GIT], [git], [])
   AC_MSG_CHECKING([revision of git repository])

   # If git executable is found in path read HEAD version
   AS_IF([test "$GIT" != ""],[
      echo "$GIT" "$srcdir" >& AS_MESSAGE_LOG_FD
      gitversion=$($GIT --git-dir=$srcdir/.git log --pretty='format:%h %ci' -1)
      gitbranch=$($GIT --git-dir=$srcdir/.git rev-parse --abbrev-ref HEAD)
      build_version="\"git $gitbranch $gitversion developer version\""
   ],[
       # Git executable not found
       build_version="\"unknown git commit\""
   ])
   AC_MSG_RESULT([$build_version])
],[
  AS_IF([test -e "$srcdir/VERSION"],[
    AC_MSG_CHECKING([revision of distributed release])
    run_gitversion=`cat $srcdir/VERSION`
    build_version="\"$run_gitversion\""
    AC_MSG_RESULT([$build_version])
  ],[
    build_version="\"unknown revision\""
  ])
])
AC_SUBST([NANOX_BUILD_VERSION],[$build_version])
AC_DEFINE_UNQUOTED([NANOX_BUILD_VERSION], [$build_version], [Build version])

# Debian release
AC_CHECK_PROG([DEB_RELEASE], [lsb_release], [$(lsb_release -sc)], [])
AC_SUBST([DEB_RELEASE])

# Set compiler default flags
: ${CFLAGS=""}
: ${CXXFLAGS=""}

# ------------------
# Check for programs
# ------------------
 AC_PROG_CC
 AC_PROG_CPP
 
 AC_PROG_CXX
 AC_PROG_CXXCPP
 
 AC_PROG_LD
 AM_PROG_AS
 
 AC_PROG_SED

 # AM_PROG_AR must be called before LT_INIT or a warning ensues
 m4_ifdef([AM_PROG_AR], [AM_PROG_AR])
 # Turn off static libraries.
 LT_INIT([disable-static])

# --------------------------------
# Compiler and flag checks
# --------------------------------

# Check for atomic builtins in GCC
AC_CHECK_GXX_LEGACY_ATOMIC_BUILTINS()
AC_CHECK_GXX_NEW_ATOMIC_BUILTINS()

# Check for header files
AC_HEADER_STDC

# Check for typedefs, structures, and compiler characteristics
AC_C_CONST

# Compiler dependent Configuration flags
AX_CONFIG_CC

## -----------------------
## Check host architecture
## -----------------------
OPTIONS=
ARCHITECTURES=

AX_CHECK_HOST_ARCH

AC_SUBST([OPTIONS])
AC_SUBST([ARCHITECTURES])

# -------------------
# Check for libraries
# -------------------

# Pthreads
AX_PTHREAD

# MPI
AX_CHECK_MPI

# GASNet
AX_CHECK_GASNET

# Memkind
AX_CHECK_MEMKIND

# Check if NVidia CUDA device has been selected 
AX_CHECK_CUDA

# Check if OpenCL device has been selected
AX_CHECK_OPENCL

# Check if SQLite3 has been selected
AX_CHECK_SQLITE3

# Check hwloc availability
AX_CHECK_HWLOC

# Check for DLB support
AX_CHECK_DLB

# Check for Zynq Xilinx DMA driver library (fpga)
AX_CHECK_XDMA

# Enable Task callback feature
AX_ENABLE_TASK_CALLBACK

# Extrae check
AX_CHECK_EXTRAE

# Check NextSim support
AC_ARG_WITH([nextsim],
           AS_HELP_STRING([--with-nextsim=dir], [Directory of NextSim installation]),
           [
              NEXTSIM_HOME="$withval"
              NEXTSIM_INC="$withval/include"
           ]
)
AC_SUBST([NEXTSIM_HOME])
AC_SUBST([NEXTSIM_INC])
AM_CONDITIONAL([instrumentation_NEXTSIM], test x"$NEXTSIM_HOME" != x)

# Check Ayudame support
AYUDAME_HOME=""
AYUDAME_INC=""
AYUDAME_LIB=""
AYUDAME_BIN=""

AC_ARG_WITH([ayudame],
           AS_HELP_STRING([--with-ayudame=dir], [Directory of Ayudame installation]),
           [
                AYUDAME_HOME="$withval"
                AYUDAME_INC="$withval/include"
                AYUDAME_LIB="$withval/lib"
                AYUDAME_BIN="$withval/bin"
           ]
)
AC_SUBST([AYUDAME_HOME])
AC_SUBST([AYUDAME_INC])
AC_SUBST([AYUDAME_LIB])
AC_SUBST([AYUDAME_BIN])

AM_CONDITIONAL([instrumentation_AYUDAME], test x"$AYUDAME_HOME" != x)

# Check for Chapel support
AC_ARG_WITH([chapel],
            AS_HELP_STRING([--with-chapel=dir], [Directory of Chapel installation]),
	    [
		chapel=yes
		CHAPEL_HOME="$withval"
	    ],
            [
		chapel=no
	    ]
)

AM_CONDITIONAL([CHAPEL_SUPPORT], [test "$chapel" = "yes"])
AC_SUBST([CHAPEL_HOME])

# Check MKL Support
AC_CHECK_LIB([mkl_sequential],[mkl_blas_dgemm],[MKL_LIBS="-lm -lmkl_core -lmkl_sequential"],[MKL_LIBS=""],[-lm -lmkl_core])
AM_CONDITIONAL([MKL_SUPPORT], test "$MKL_LIBS"x != x )
AC_SUBST([MKL_LIBS])


# mcc support
AC_ARG_WITH([mcc],
AS_HELP_STRING([--with-mcc=dir], [Directory of mcc compiler]),
	[
      mcc_valid=yes
		MCC="${withval}/bin/mcc"
		MCXX="${withval}/bin/mcxx"
	],
	[
      mcc_valid=no
	])

AC_SUBST([mcc_valid])
AC_SUBST([MCC])
AC_SUBST([MCXX])


# Build versions
AX_BUILD_VERSIONS

# Allocator support
AC_MSG_CHECKING([if Nanos++ Allocator has been enabled])
AC_ARG_ENABLE([allocator], [AS_HELP_STRING([--enable-allocator], [Enables Allocator module])],
      [], dnl Implicit: enable_allocator=$enableval
      [enable_allocator="no"])
AC_MSG_RESULT([$enable_allocator])
AS_IF([test "$enable_allocator" = yes],[
      AC_DEFINE([NANOS_USE_ALLOCATOR],[1],[Specifies whether Nanos++ allocator has been enabled])
])

# Memtracker support
AC_MSG_CHECKING([if Nanos++ Memtracker has been enabled])
AC_ARG_ENABLE([memtracker], [AS_HELP_STRING([--enable-memtracker], [Enables Memtracker module])],
      [], dnl Implicit: enable_memtracker=$enableval
      [enable_memtracker="no"])
AC_MSG_RESULT([$enable_memtracker])
AS_IF([test "$enable_memtracker" = xyes],[
      AC_DEFINE([NANOS_MEMTRACKER_ENABLED],[1],[Specifies whether Nanos++ memtracker module has been enabled])
      AC_SUBST([NANOS_MEMTRACKER_ENABLED], [NANOS_MEMTRACKER_ENABLED])
], [
      AC_SUBST([NANOS_MEMTRACKER_ENABLED], [NO_NANOS_MEMTRACKER_ENABLED])
])

# Task-level resiliency support
AC_MSG_CHECKING([if task resiliency is enabled])
AC_ARG_ENABLE([resiliency],[AS_HELP_STRING([--enable-resiliency], [Enables task-level resiliency])],
              [enable_resiliency=$enableval],[enable_resiliency=no])
AC_MSG_RESULT([$enable_resiliency])
AS_IF([test $enable_resiliency = yes],[
  AX_CHECK_COMPILE_FLAG([-fnon-call-exceptions],
    [],
    [AC_MSG_ERROR([resiliency mechanism depends on using compiler flag -fnon-call-exceptions])],
    [-Werror])

  AC_DEFINE([NANOS_RESILIENCY_ENABLED],[],[Indicates whether resiliency features should be used or not.])
  AC_SUBST([NANOS_RESILIENCY_ENABLED],[NANOS_RESILIENCY_ENABLED])
  resiliency_flags=-fnon-call-exceptions
], [
  AC_SUBST([NANOS_RESILIENCY_ENABLED],[NO_NANOS_RESILIENCY_ENABLED])
])
AC_SUBST([enable_resiliency])
AC_SUBST([resiliency_flags])

# Generate plugin list
PLUGINS="plugins pms arch/$OS"
for arch in $ARCHITECTURES; do
  PLUGINS+=" arch/$arch"
done
AC_SUBST([PLUGINS])

# Output files
AC_CONFIG_HEADERS([config.h])

AC_CONFIG_FILES([
                 Makefile
                 doc/Makefile
                 doc/doxy-nanox
                 scripts/Makefile
                 src/Makefile
                 src/core/Makefile
                 src/support/Makefile
                 src/apis/Makefile
                 src/pms/Makefile
                 src/plugins/Makefile
                 src/arch/Makefile
                 src/arch/unix-os/Makefile
                 src/arch/smp/Makefile
                 src/arch/cluster/Makefile
                 src/arch/fpga/Makefile
                 src/arch/spu/Makefile
                 src/arch/gpu/Makefile
                 src/arch/opencl/Makefile
                 src/arch/mpi/Makefile
                 src/arch/mpi/nanox_compiler
                 src/utils/Makefile
                 scripts/offload_slave_launch.sh
                 tests/Makefile
                 nanox-config.h
       ])

AC_CONFIG_FILES([tests/gens/api-generator], [chmod +x tests/gens/api-generator])
AC_CONFIG_FILES([tests/gens/api-omp-generator], [chmod +x tests/gens/api-omp-generator])
AC_CONFIG_FILES([tests/gens/core-generator], [chmod +x tests/gens/core-generator])
AC_CONFIG_FILES([tests/gens/mcc-openmp-generator], [chmod +x tests/gens/mcc-openmp-generator])
AC_CONFIG_FILES([tests/gens/mcc-ompss-generator], [chmod +x tests/gens/mcc-ompss-generator])
AC_CONFIG_FILES([tests/gens/opencl-generator], [chmod +x tests/gens/opencl-generator])
AC_CONFIG_FILES([tests/gens/resiliency-generator], [chmod +x tests/gens/resiliency-generator])

AC_OUTPUT

gcc_builtins_used="unknown"
AS_IF([test "$new_gcc_builtins" = yes],[
  gcc_builtins_used="new gcc __atomic builtins"
],[
  gcc_builtins_used="legacy gcc __sync builtins"
])

AC_DEFUN([ax_check_enabled],[
  AS_IF([test x$1 == xyes],[
    echo -n "enabled"
  ],[
    echo -n "disabled"
  ])
])

AS_ECHO(["
Configuration summary of Nanos++
================================

Host arch:                $SMP_ARCH
User level threading:     $ult_support
Configured architectures: $ARCHITECTURES
Configured versions:      $VERSIONS
Extra options:            $OPTIONS
GCC atomics:              $gcc_builtins_used
Memory tracker:           $(ax_check_enabled([$enable_memtracker]))
Memory allocator:         $(ax_check_enabled([$enable_allocator]))
Task resiliency:          $(ax_check_enabled([$enable_resiliency]))"])

AS_IF([test "$gasnet_available_conduits" != ""],[
   AS_ECHO(["\
Cluster/GASNet conduits:  $gasnet_available_conduits"])
])

