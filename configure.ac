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

# AM_PROG_AR must be called before LT_INIT or a warning ensues
m4_ifdef([AM_PROG_AR], [AM_PROG_AR])
# Turn off static libraries.
LT_INIT([disable-static])

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

# Set compiler default flags
: ${CFLAGS=""}
: ${CXXFLAGS=""}

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

# Check for programs
AC_PROG_CPP
AC_PROG_CC([$CC])
AC_PROG_CXX([$CXX])
AC_PROG_LD([$LD])
AM_PROG_CC_C_O
AM_PROG_AS

AC_PROG_SED

# Flags & Compiler dependent stuff
AX_CONFIG_CC

# Check for libraries

# Check for header files
AC_HEADER_STDC

# Check for typedefs, structures, and compiler characteristics
AC_C_CONST

# Check for library functions
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

ARCHITECTURES=
OPTIONS=
OS=

AC_SUBST([ARCHITECTURES])
AC_SUBST([OS])
AC_SUBST([OPTIONS])
AC_SUBST([SMP_ARCH])

AC_CHECK_SIZEOF([size_t])

#AC_MSG_CHECKING([for architecture])
AC_ARG_ENABLE([ult],
  AS_HELP_STRING([--disable-ult], [Disables user level threads]),
  [smp_ult=no],
  [smp_ult=yes]
)

# Use these for flags that depend on the host arch
host_dep_CPPFLAGS=
host_dep_CXXFLAGS=
host_dep_LDFLAGS=

AC_SUBST([host_dep_CPPFLAGS])
AC_SUBST([host_dep_CXXFLAGS])
AC_SUBST([host_dep_LDFLAGS])

case $host in
  x86_64-k1om-linux*) # Intel MIC (KNF/ KNC)
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"
    SMP_ARCH=x86-64 # User level threads implementation for x86_64 might not be compatible with MIC

    AX_CHECK_MIC_SUPPORT
    ;;
  i?86-*-linux*)
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"
    SMP_ARCH=x86
    ;;
  x86_64-*-linux*)
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"
    SMP_ARCH=x86-64
    ;;
  ia64-*-linux*)
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"
    SMP_ARCH=ia64
    ;;
  arm*-*-gnueabi*)
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"
    SMP_ARCH=armv71
    ;;
  aarch64-*-linux-gnu)
    smp_ult=no
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"
    SMP_ARCH=aarch64
    ;;
  powerpc64-bgq-linux-gnu)
    # BlueGene/Q
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"

    SMP_ARCH=ppc64

    AX_CHECK_BGQ

    ;;
  powerpc64??-*-linux* )
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"

    if test $ac_cv_sizeof_size_t = 8 ; then
        SMP_ARCH=ppc64_v2
    fi
    #FIXME ULT not supported yet
    smp_ult=no
    ;;

  powerpc-*-linux* | powerpc64-*-linux* )
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"

    if test $ac_cv_sizeof_size_t = 8 ; then
        SMP_ARCH=ppc64
    else
        SMP_ARCH=ppc32
    fi

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

	   NANOS_CONFIG_LIBDIRS="NANOS_CONFIG_LIBDIRS -L$CELLSDK_LIB"
	   NANOS_CONFIG_LIBS="NANOS_CONFIG_LIBS -lspe2"
        fi
      fi
      if test x$spu_valid = xyes; then
        ARCHITECTURES="$ARCHITECTURES spu"
      fi
    fi
    ;;
  *)
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"
    SMP_ARCH=unknown
    smp_ult=no
    ;;
esac

#check for cluster enable option
AC_MSG_CHECKING([if Cluster architecture was requested])
AC_ARG_ENABLE([cluster],
    AS_HELP_STRING([--enable-cluster], [Enables cluster architecture]),
    [
        if test x$enableval = xyes -o x$enableval = x;
        then
            cluster_enabled="yes"
            AC_MSG_RESULT([yes])
        else if test x$enableval = xno;
            then
                AC_MSG_RESULT([no])
                cluster_enabled="no"
             else
                AC_MSG_ERROR([This option can only be given 'yes' or 'no' values])
             fi
        fi
    ],
    [
        cluster_enabled="no"
        AC_MSG_RESULT([yes])
    ]
)
with_mpi_conduit="no"
with_udp_conduit="no"
with_smp_conduit="no"
with_ibv_conduit="no"
with_mpi_conduit_dbg="no"
with_udp_conduit_dbg="no"
with_smp_conduit_dbg="no"
with_ibv_conduit_dbg="no"
if test x$cluster_enabled = xyes ; then

    m4_include(lx_find_mpi.m4)

    CFLAGS="$CFLAGS -DCLUSTER_DEV"
    CXXFLAGS="$CXXFLAGS -DCLUSTER_DEV"
    ARCHITECTURES="$ARCHITECTURES cluster"

    AC_ARG_WITH([gasnet],
          AS_HELP_STRING([--with-gasnet=dir], [Directory of GASNet installation]),
          [
    	GASNET_DIR="${withval}"
          ],
          [
    	GASNET_DIR=/opt/gasnet
          ])

    AC_ARG_WITH([gasnet-dbg],
          AS_HELP_STRING([--with-gasnet-dbg=dir], [Directory of GASNet installation with debug options enabled]),
          [
    	GASNET_DBG_DIR="${withval}"
          ],
          [
    	GASNET_DBG_DIR=$GASNET_DIR
          ])

    AC_SUBST([GASNET_DIR])
    AC_SUBST([GASNET_DBG_DIR])
    AC_SUBST([MPI_CLDFLAGS])
    CONDUITS=`ls -d $GASNET_DIR/include/*-conduit | sed -e "s#$GASNET_DIR/include/##" | cut -d - -f 1`
    for this_conduit in $CONDUITS ; do
        if test x$this_conduit = xmpi ; then
            with_mpi_conduit="yes"
            #AC_LANG_PUSH([C++])
            export MP_CXX=gcc
            LX_FIND_MPI()
            #AC_LANG_POP([C++])
            #echo have_C_mpi is $have_C_mpi : MPICC is $MPICC / MPI_CFLAGS is $MPI_CFLAGS / MPI_CLDFLAGS is $MPI_CLDFLAGS
            #echo have_CXX_mpi is $have_CXX_mpi : MPICXX is $MPICXX / MPI_CXXFLAGS is $MPI_CXXFLAGS / MPI_CXXLDFLAGS is $MPI_CXXLDFLAGS
        fi

        if test x$this_conduit = xudp ; then
            with_udp_conduit="yes"
        fi

        if test x$this_conduit = xsmp ; then
            with_smp_conduit="yes"
        fi

        if test x$this_conduit = xibv ; then
            with_ibv_conduit="yes"
        fi
    done
    if test x$GASNET_DIR != x$GASNET_DBG_DIR ; then
       CONDUITS_DBG=`ls -d $GASNET_DBG_DIR/include/*-conduit | sed -e "s#$GASNET_DBG_DIR/include/##" | cut -d - -f 1`
       for this_conduit in $CONDUITS_DBG ; do
           if test x$this_conduit = xmpi ; then
               with_mpi_conduit_dbg="yes"
               #AC_LANG_PUSH([C++])
               export MP_CXX=gcc
               LX_FIND_MPI()
               #AC_LANG_POP([C++])
               #echo have_C_mpi is $have_C_mpi : MPICC is $MPICC / MPI_CFLAGS is $MPI_CFLAGS / MPI_CLDFLAGS is $MPI_CLDFLAGS
               #echo have_CXX_mpi is $have_CXX_mpi : MPICXX is $MPICXX / MPI_CXXFLAGS is $MPI_CXXFLAGS / MPI_CXXLDFLAGS is $MPI_CXXLDFLAGS
           fi

           if test x$this_conduit = xudp ; then
               with_udp_conduit_dbg="yes"
           fi

           if test x$this_conduit = xsmp ; then
               with_smp_conduit_dbg="yes"
           fi

           if test x$this_conduit = xibv ; then
               with_ibv_conduit_dbg="yes"
           fi
       done
    fi
fi
AM_CONDITIONAL([CONDUIT_MPI], [test x$with_mpi_conduit = xyes])
AM_CONDITIONAL([CONDUIT_UDP], [test x$with_udp_conduit = xyes])
AM_CONDITIONAL([CONDUIT_SMP], [test x$with_smp_conduit = xyes])
AM_CONDITIONAL([CONDUIT_IBV], [test x$with_ibv_conduit = xyes])
AM_CONDITIONAL([CONDUIT_DBG_MPI], [test x$with_mpi_conduit_dbg = xyes])
AM_CONDITIONAL([CONDUIT_DBG_UDP], [test x$with_udp_conduit_dbg = xyes])
AM_CONDITIONAL([CONDUIT_DBG_SMP], [test x$with_smp_conduit_dbg = xyes])
AM_CONDITIONAL([CONDUIT_DBG_IBV], [test x$with_ibv_conduit_dbg = xyes])

# Pthreads
AX_PTHREAD

#MPI Support
AX_CHECK_MPI

# Check if NVidia CUDA device has been selected 
AX_CHECK_CUDA

# Check if OpenCL device has been selected
AX_CHECK_OPENCL

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
		MCC=""
	])

AC_SUBST([mcc_valid])
AC_SUBST([MCC])
AC_SUBST([MCXX])

AC_MSG_CHECKING([for architecture])

# Add architecture libraries
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

VERSIONS=""

# Enable debug version
AC_MSG_CHECKING([if debug version is enabled])
AC_ARG_ENABLE([debug],
  [AS_HELP_STRING([--disable-debug], [Disables generation of debug version])],
  [enable_debug=$enableval],[enable_debug=yes])

if test x$enable_debug = xyes; then
  AC_MSG_RESULT([yes])
  VERSIONS+=" debug"

  debug_CPPFLAGS="-DNANOS_DEBUG_ENABLED"
  debug_CXXFLAGS="-O0 -g2 $no_inline_flag"
else
  AC_MSG_RESULT([no])
fi

AM_CONDITIONAL([DEBUG_VERSION], [test x$enable_debug = xyes])
AC_SUBST([debugdir],['${libdir}/debug'])

# Enable instrumentation version
AC_MSG_CHECKING([if instrumentation version is enabled])
AC_ARG_ENABLE([instrumentation],
  [AS_HELP_STRING([--disable-instrumentation], [Disables generation of instrumentation version])],
  [enable_instrumentation=$enableval],[enable_instrumentation=yes])

if test x$enable_instrumentation = xyes; then
  AC_MSG_RESULT([yes])
  VERSIONS+=" instrumentation"

  instrumentation_CPPFLAGS="-DNANOS_INSTRUMENTATION_ENABLED"
  instrumentation_CXXFLAGS="-O3"
else
  AC_MSG_RESULT([no])
fi

AM_CONDITIONAL([INSTRUMENTATION_VERSION], [test x$enable_instrumentation = xyes])
AC_SUBST([instrumentationdir],['${libdir}/instrumentation'])

# Enable instrumentation-debug version
AC_MSG_CHECKING([if instrumentation-debug version was enabled])
AC_ARG_ENABLE([instrumentation-debug],
  [AS_HELP_STRING([--enable-instrumentation-debug], [Enables generation of instrumentation-debug version])],
  [enable_instrumentation_debug=$enableval],[enable_instrumentation_debug=no])

if test x$enable_instrumentation_debug = xyes; then
  AC_MSG_RESULT([yes])
  VERSIONS+=" instrumentation-debug"

  instrumentation_debug_CPPFLAGS="-DNANOS_DEBUG_ENABLED -DNANOS_INSTRUMENTATION_ENABLED"
  instrumentation_debug_CXXFLAGS="-O0 -g2 $no_inline_flag"
else
  AC_MSG_RESULT([no])
fi

AM_CONDITIONAL([INSTRUMENTATION_DEBUG_VERSION], [test x$enable_instrumentation_debug = xyes])
AC_SUBST([instrumentation_debugdir],['${libdir}/instrumentation-debug'])

# Enable performance version
AC_MSG_CHECKING([if performance version is enabled])
AC_ARG_ENABLE([performance],
  AS_HELP_STRING([--disable-performance], [Disables generation of performance version])],
  [enable_performance=$enableval],[enable_performance=yes])

if test x$enable_performance = xyes; then
  AC_MSG_RESULT([yes])
  VERSIONS+=" performance"

  performance_CPPFLAGS=
  performance_CXXFLAGS="-O3"
else
  AC_MSG_RESULT([no])
fi

AM_CONDITIONAL([PERFORMANCE_VERSION], [test x$enable_performance = xyes])
AC_SUBST([performancedir],['${libdir}/performance'])

if test "x$VERSIONS" = x;
then
	AC_MSG_ERROR([At least one version needs to be compiled])
fi
AC_SUBST([VERSIONS])

AC_SUBST([debug_CPPFLAGS])
AC_SUBST([debug_CXXFLAGS])
AC_SUBST([instrumentation_CPPFLAGS])
AC_SUBST([instrumentation_CXXFLAGS])
AC_SUBST([instrumentation_debug_CPPFLAGS])
AC_SUBST([instrumentation_debug_CXXFLAGS])
AC_SUBST([performance_CPPFLAGS])
AC_SUBST([performance_CXXFLAGS])

# Check for atomic builtins in GCC
AC_CHECK_GXX_ATOMIC_BUILTINS()

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

AM_CONDITIONAL([INSTRUMENTATION_EXTRAE], test x"$MPITRACE_HOME" != x)

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
AM_CONDITIONAL([INSTRUMENTATION_NEXTSIM], test x"$NEXTSIM_HOME" != x)

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

AM_CONDITIONAL([INSTRUMENTATION_AYUDAME], test x"$AYUDAME_HOME" != x)

# Check hwloc availability
AX_CHECK_HWLOC

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

# Task-level resiliency support
AC_MSG_CHECKING([if task resiliency is enabled])
AC_ARG_ENABLE([resiliency],[AS_HELP_STRING([--enable-resiliency], [Enables task-level resiliency])],
              [enable_resiliency=$enableval],[enable_resiliency=no])
AC_MSG_RESULT([$enable_resiliency])
if test x$enable_resiliency = xyes; then
  AX_CHECK_COMPILE_FLAG([-fnon-call-exceptions],
    [],
    [AC_MSG_ERROR([resiliency mechanism depends on using compiler flag -fnon-call-exceptions])],
    [-Werror])

  AC_DEFINE([NANOS_RESILIENCY_ENABLED],[],[Indicates whether resiliency features should be used or not.])
  extra_CXXFLAGS=-fnon-call-exceptions
fi
AC_SUBST([enable_resiliency])

AC_SUBST([extra_CPPFLAGS])
AC_SUBST([extra_CXXFLAGS])
AC_SUBST([extra_LDFLAGS])
AC_SUBST([extra_LIBS])

# Generate plugin list
PLUGINS="plugins pms/openmp pms/chapel"
for arch in $ARCHITECTURES; do
  PLUGINS+=" arch/$arch"
done
AC_SUBST([PLUGINS])

# Output files
AC_CONFIG_HEADERS([config.h])

AC_CONFIG_FILES([Makefile
                 src/Makefile
                 src/core/Makefile
                 src/support/Makefile
                 src/plugins/Makefile
                 src/apis/Makefile
                 src/pms/Makefile
                 src/arch/Makefile
                 src/arch/unix-os/Makefile
                 src/arch/smp/Makefile
                 src/arch/fpga/Makefile
                 src/arch/spu/Makefile
                 src/arch/gpu/Makefile
                 src/arch/cluster/Makefile
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
                 tests/gens/mixed-generator
                 tests/gens/mpi-offload-generator
                 tests/gens/nanox-creation-generator
                 tests/gens/resiliency-generator
                 doc/Makefile
                 doc/doxy-nanox
       ])

AC_OUTPUT

echo ""
echo "Configuration summary of Nanos++"
echo "================================"
echo ""
echo "Host arch:               " $SMP_ARCH
echo "User level threading:    " $smp_ult
echo "Configured architectures:" $ARCHITECTURES
echo "Configured versions:     " $VERSIONS
echo "Extra options:           " $OPTIONS
if test x$memtracker_enabled = xyes ; then
   echo "Memory tracker:           enabled"
else
   echo "Memory tracker:           disabled"
fi
if test x$allocator_enabled = xyes ; then
   echo "Memory allocator:         enabled"
else
   echo "Memory allocator:         disabled"
fi
if test "x$resiliency" = "xyes" ; then
   echo "Task resiliency:          enabled"
else
   echo "Task resiliency:          disabled"
fi
if test x$cluster_enabled = xyes ; then
   echo "With cluster/gasnet conduits:" $CONDUITS
   if test x$have_C_mpi = xyes ; then
      echo "MPI_CLDFLAGS: " $MPI_CLDFLAGS
   fi
   if test x$GASNET_DIR != x$GASNET_DBG_DIR ; then
      echo "GASNet was configured also with a debug installation, with these conduits: " $CONDUITS_DBG
   fi
fi

