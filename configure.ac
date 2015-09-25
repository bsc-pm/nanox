AC_INIT([nanox], [0.9a], [pm-tools@bsc.es])
AC_LANG([C++])
AC_CONFIG_SRCDIR([src/core/system.cpp])
AC_DEFINE_UNQUOTED([NANOX_CONFIGURE_ARGS], ["$0 $@"], [Configure line])

# Environment
AC_CANONICAL_BUILD
AC_CANONICAL_SYSTEM
AC_CANONICAL_TARGET

# Automake initialization
AM_INIT_AUTOMAKE([-Wall dist-bzip2 -Wno-portability subdir-objects])
AC_CONFIG_MACRO_DIR([m4])

dnl Enable silent rules if available
supported_silent_rules=1
m4_ifdef([AM_SILENT_RULES],
         [AM_SILENT_RULES([yes])],
         [supported_silent_rules=0])

AM_CONDITIONAL([SUPPORTED_SILENT_RULES], test x$supported_silent_rules = x1)
 
dnl Versioning things
dnl Make sure build_version is double-quoted!
build_version="\"unknown revision\""
if test -d $srcdir/.git;
then
   AC_PATH_PROG([GIT], [git], [])
   AC_MSG_CHECKING([revision of git repository])
   run_git=""
   if test x"$GIT" != x;
   then
      echo "$GIT" "$srcdir" >& AS_MESSAGE_LOG_FD
      run_gitversion=$($GIT --git-dir=$srcdir/.git log --pretty='format:%h %ci' -1)
      run_gitbranch=$($GIT --git-dir=$srcdir/.git rev-parse --abbrev-ref HEAD)
      build_version="\"git $run_gitbranch $run_gitversion developer version\""
   fi
   if test x"$GIT" = x -o x"$run_gitversion" = x;
   then
       build_version="\"unknown git commit\""
   fi
   AC_MSG_RESULT([$build_version])
elif test -e $srcdir/VERSION;
then
     AC_MSG_CHECKING([revision of distributed release])
     run_gitversion=`cat $srcdir/VERSION`
     build_version="\"$run_gitversion\""
     AC_MSG_RESULT([$build_version])
fi
AC_DEFINE_UNQUOTED([NANOX_BUILD_VERSION], [$build_version], [Build version])

# Debian release
AC_CHECK_PROG([DEB_RELEASE], [lsb_release], [$(lsb_release -sc)], [])
AC_SUBST([DEB_RELEASE])

# Allocator support
AC_MSG_CHECKING([if Nanos++ Allocator has been enabled])
AC_ARG_ENABLE([allocator], [AS_HELP_STRING([--enable-allocator], [Enables Allocator module])])
AC_MSG_RESULT([$enable_allocator])
if test x$enable_allocator = xyes; then
	AC_DEFINE([NANOS_USE_ALLOCATOR],[1],[Specifies whether Nanos++ allocator has been enabled or not])
fi

# Memtracker support
AC_MSG_CHECKING([if Nanos++ Memtracker has been enabled])
AC_ARG_ENABLE([memtracker],[AS_HELP_STRING([--enable-memtracker], [Enables Memtracker module])])
AC_MSG_RESULT([$enable_memtracker])
if test x$enable_memtracker = xyes; then
	AC_DEFINE([NANOS_MEMTRACKER_ENABLED],[1],[Specifies whether Nanos++ memtracker module has been enabled or not])
fi

# User level threads
# System dependent. Architecture support will make a compatibility check.
AC_ARG_ENABLE([ult],
  AS_HELP_STRING([--disable-ult], [Disables user level threads]),
  [smp_ult=${enableval}],
  [smp_ult=yes]
)

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
OS=
OPTIONS=
SMP_ARCH=
ARCHITECTURES=

AX_CHECK_HOST_ARCH

AC_SUBST([OS])
AC_SUBST([OPTIONS])
AC_SUBST([SMP_ARCH])
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

# Cluster Support
AX_CHECK_CLUSTER

# Memkind
AX_CHECK_MEMKIND

# Check if NVidia CUDA device has been selected 
AX_CHECK_CUDA

# Check if OpenCL device has been selected
AX_CHECK_OPENCL

# Check hwloc availability
AX_CHECK_HWLOC

# Extrae check
MPITRACE_HOME=""
MPITRACE_INC=""
MPITRACE_LIB=""
MPITRACE_BIN=""

AC_ARG_WITH([extrae],
           AS_HELP_STRING([--with-extrae=dir], [Directory of Extrae installation]),
           [
                extrae_version_h=$withval/include/extrae_version.h
                if test -e $extrae_version_h ;
                then
                   extrae_version_string=`sed -ne '/^\s*#\s*define\s\+EXTRAE_VERSION .*$/p' $extrae_version_h`
                   extrae_version=`echo $extrae_version_string | sed -e 's/#define EXTRAE_VERSION EXTRAE_VERSION_NUMBER(\([0-9]*\),\([0-9]*\),\([0-9]*\).*$/\1\2\3/'`
                   if test "$extrae_version" -ge 240;
                   then
                      MPITRACE_HOME="$withval"
                      MPITRACE_INC="$withval/include"
                      MPITRACE_LIB="$withval/lib"
                      if test -d "$MPITRACE_HOME/lib64";
                      then
                         MPITRACE_LIB="$MPITRACE_HOME/lib64"
                      fi
                      MPITRACE_BIN="$withval/bin"
                      AC_MSG_RESULT([checking if Extrae library is compatible... yes])
                   else
                      AC_MSG_ERROR([checking if Extrae library is compatible... no (Extrae > 2.4 needed)])
                   fi
                else
                   AC_MSG_ERROR([checking if Extrae library is compatible... no (Extrae > 2.4 needed)])
                fi
           ]
)

AC_SUBST([MPITRACE_HOME])
AC_SUBST([MPITRACE_INC])
AC_SUBST([MPITRACE_LIB])
AC_SUBST([MPITRACE_BIN])

AM_CONDITIONAL([instrumentation_EXTRAE], test x"$MPITRACE_HOME" != x)

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

# Check for DLB support
AC_MSG_CHECKING([for DLB support])
AC_ARG_WITH([dlb],
   AS_HELP_STRING([--with-dlb=dir], [Directory of DLB installation]),
   [
      dlb_path="$withval"
   ],
   [
      dlb_path="not_set"
   ]
)

if test ${dlb_path} != "no" -a ${dlb_path} != "not_set" ; then
   DLB_INC="$dlb_path/include"
   DLB_LIB="$dlb_path/lib"
   if test ! -d ${dlb_path} -o ! -f ${DLB_INC}/DLB_interface.h -o ! -f ${DLB_LIB}/libdlb.so ; then
      AC_MSG_ERROR([Attention! You have passed an invalid DLB location.])
   fi
   dlb=yes
   AC_DEFINE([DLB],[],[Enables DLB support.])
else
   dlb=no
fi
AC_MSG_RESULT([${dlb}])

AM_CONDITIONAL([DLB], [test x"$dlb" = "xyes"])
AC_SUBST([DLB_INC])
AC_SUBST([DLB_LIB])

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

AC_MSG_CHECKING([for architecture])

# Add architecture libraries
NANOS_CONFIG_LIBDIRS=
NANOS_CONFIG_LIBS=
NANOS_CONFIG_LIBS_PERFO=
NANOS_CONFIG_LIBS_DEBUG=
NANOS_CONFIG_LIBS_INSTR=
AC_SUBST([NANOS_CONFIG_LIBDIRS])
AC_SUBST([NANOS_CONFIG_LIBS_PERFO])
AC_SUBST([NANOS_CONFIG_LIBS_DEBUG])
AC_SUBST([NANOS_CONFIG_LIBS_INSTR])
AC_SUBST([NANOS_CONFIG_LIBS_INSTRDEBUG])

for arch in $ARCHITECTURES; do
   if test x"$arch" != x"mpi"; then
      NANOS_CONFIG_LIBS_PERFO="$NANOS_CONFIG_LIBS_PERFO $NANOS_CONFIG_LIBS \$(abs_top_builddir)/src/arch/$arch/performance/lib$arch.la"
      NANOS_CONFIG_LIBS_DEBUG="$NANOS_CONFIG_LIBS_DEBUG $NANOS_CONFIG_LIBS \$(abs_top_builddir)/src/arch/$arch/debug/lib$arch.la"
      NANOS_CONFIG_LIBS_INSTR="$NANOS_CONFIG_LIBS_INSTR $NANOS_CONFIG_LIBS \$(abs_top_builddir)/src/arch/$arch/instrumentation/lib$arch.la"
      NANOS_CONFIG_LIBS_INSTRDEBUG="$NANOS_CONFIG_LIBS_INSTRDEBUG $NANOS_CONFIG_LIBS \$(abs_top_builddir)/src/arch/$arch/instrumentation-debug/lib$arch.la"
   fi
done
NANOS_CONFIG_LIBS_PERFO="$NANOS_CONFIG_LIBS_PERFO \$(abs_top_builddir)/src/arch/$OS/performance/libos.la \$(abs_top_builddir)/src/support/performance/libsupport.la"
NANOS_CONFIG_LIBS_DEBUG="$NANOS_CONFIG_LIBS_DEBUG \$(abs_top_builddir)/src/arch/$OS/debug/libos.la \$(abs_top_builddir)/src/support/debug/libsupport.la"
NANOS_CONFIG_LIBS_INSTR="$NANOS_CONFIG_LIBS_INSTR \$(abs_top_builddir)/src/arch/$OS/instrumentation/libos.la \$(abs_top_builddir)/src/support/instrumentation/libsupport.la"
NANOS_CONFIG_LIBS_INSTRDEBUG="$NANOS_CONFIG_LIBS_INSTRDEBUG \$(abs_top_builddir)/src/arch/$OS/instrumentation-debug/libos.la \$(abs_top_builddir)/src/support/instrumentation-debug/libsupport.la"

AM_CONDITIONAL([SMP_SUPPORTS_ULT],[test $smp_ult = yes])

AC_MSG_RESULT([$ARCHITECTURES/$OS])

# Build versions
AX_BUILD_VERSIONS

# Task-level resiliency support
AC_MSG_CHECKING([if task resiliency is enabled])
AC_ARG_ENABLE([resiliency],[AS_HELP_STRING([--enable-resiliency], [Enables task-level resiliency])],
              [enable_resiliency=$enableval],[enable_resiliency=no])
AC_MSG_RESULT([$enable_resiliency])
if test $enable_resiliency = yes; then
  AX_CHECK_COMPILE_FLAG([-fnon-call-exceptions],
    [],
    [AC_MSG_ERROR([resiliency mechanism depends on using compiler flag -fnon-call-exceptions])],
    [-Werror])

  AC_DEFINE([NANOS_RESILIENCY_ENABLED],[],[Indicates whether resiliency features should be used or not.])
  resiliency_flags=-fnon-call-exceptions
fi
AC_SUBST([enable_resiliency])
AC_SUBST([resiliency_flags])

# Generate plugin list
PLUGINS="plugins pms/openmp pms/chapel"
for arch in $ARCHITECTURES; do
  PLUGINS+=" arch/$arch"
done
AC_SUBST([PLUGINS])

# Output files
AC_CONFIG_HEADERS([config.h])

AC_CONFIG_FILES([
                 Makefile
                 src/common
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
                 src/utils/Makefile
                 scripts/Makefile
                 tests/Makefile
                 tests/gens/Makefile
                 tests/gens/api-generator
                 tests/gens/api-omp-generator
                 tests/gens/core-generator
                 tests/gens/gpu-generator
                 tests/gens/mcc-openmp-generator
                 tests/gens/mcc-ompss-generator
                 tests/gens/mixed-generator
                 tests/gens/nanox-creation-generator
                 tests/gens/resiliency-generator
                 doc/Makefile
                 doc/doxy-nanox
       ])

AC_OUTPUT

gcc_builtins_used="unknown"
if test "$new_gcc_builtins" = yes;
then
    gcc_builtins_used="new gcc __atomic builtins"
else
    gcc_builtins_used="legacy gcc __sync builtins"
fi

AS_ECHO(["
Configuration summary of Nanos++
================================

Host arch:                $SMP_ARCH
User level threading:     $smp_ult
Configured architectures: $ARCHITECTURES
Configured versions:      $VERSIONS
Extra options:            $OPTIONS
GCC atomics:              $gcc_builtins_used
Memory tracker:           $enable_memtracker
Memory allocator:         $enable_allocator
Task resiliency:          $resiliency"])

if test x$cluster_enabled = xyes ; then
   AS_ECHO(["\
Cluster/GASNet conduits:  $gasnet_available_conduits"])
fi
