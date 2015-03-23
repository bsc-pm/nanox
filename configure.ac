AC_INIT([nanox], [0.9a], [pm-tools@bsc.es])
AC_CONFIG_SRCDIR([src/core/system.cpp])

configure_orig_arguments="$@"

# New comment

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
      run_gitversion=`"$GIT" --git-dir=$srcdir/.git show "--pretty=format:%h %ci" HEAD | head -n 1`
      run_gitbranch=`"$GIT" --git-dir=$srcdir/.git branch | grep ^* | sed s/*\ //g`
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

nanox_config_args='"'$(echo "$0 $configure_orig_arguments")'"'
AC_DEFINE_UNQUOTED([NANOX_CONFIGURE_ARGS], [$nanox_config_args], [Configure line])

# Debian release
AC_CHECK_PROG([DEB_RELEASE], [lsb_release], [$(lsb_release -sc)], [])
AC_SUBST([DEB_RELEASE])

# Set compiler default flags
: ${CFLAGS=""}
: ${CXXFLAGS=""}

# Allocator support
AC_MSG_CHECKING([if Nanos++ Allocator has been enabled])
AC_ARG_ENABLE([allocator],
  AS_HELP_STRING([--enable-allocator], [Enables Allocator module]),
  [
    allocator_enabled=$enableval
    AC_MSG_RESULT([$enableval])
  ],
  [
    allocator_enabled=no
    AC_MSG_RESULT([no])
  ]
)
if test x$allocator_enabled = xyes; then
	AC_DEFINE([NANOS_USE_ALLOCATOR],[1],[Specifies whether Nanos++ allocator has been enabled or not])
fi

# Memtracker support
AC_MSG_CHECKING([if Nanos++ Memtracker has been enabled])
AC_ARG_ENABLE([memtracker],
  AS_HELP_STRING([--enable-memtracker], [Enables Memtracker module]),
  [
    memtracker_enabled=$enableval
    AC_MSG_RESULT([$enableval])
  ],
  [
    memtracker_enabled=no
    AC_MSG_RESULT([no])
  ]
)
if test x$memtracker_enabled = xyes; then
	AC_DEFINE([NANOS_MEMTRACKER_ENABLED],[1],[Specifies whether Nanos++ memtracker module has been enabled or not])
fi

# Check for programs

# Check for architecture
ARCHITECTURES=
OS=

# Check for programs
AC_PROG_CPP
AC_PROG_CC([$CC])
AC_PROG_CXX([$CXX])
AC_PROG_LD([$LD])
AM_PROG_CC_C_O
AM_PROG_AS

AC_PROG_SED

# Flags & Compiler dependent stuff
if test x$CC=xicc -o x$CXX=xicpc; then
	cc_intel=yes;
fi

if test x$ac_cv_cxx_compiler_gnu = xyes; then
   cc_gnu_compatible=yes
fi

if test x$ac_cv_prog_cc_xlc_c_o = xyes; then
   cc_xlc=yes
fi
if test x$ac_cv_prog_cc_xlc_r_c_o = xyes; then
   cc_xlc=yes
fi

if test x$use_bgq = xyes; then
    if test x$use_gnu = xyes; then
        AC_SUBST([BGQ_DYNAMIC_LINK],[-dynamic])
    fi
    if test x$use_xlc = xyes; then
        AC_SUBST([BGQ_DYNAMIC_LINK],[-qnostaticlink])
    fi
fi

AM_CONDITIONAL([GNU_C], [test x$use_gnu = xyes])
AM_CONDITIONAL([IBM_C], [test x$use_xlc = xyes])
AM_CONDITIONAL([INTEL_C], [test x$cc_intel = xyes])

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

OPTIONS=

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

#MPI Support
#AX_CHECK_MPI

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
	],
	[
      mcc_valid=no
		MCC=""
	])

AC_SUBST([mcc_valid])
AC_SUBST([MCC])

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

USED_VERSION=""
VERSIONS=""

# Enable instrumentation-debug version
instrumentation_debug_disabled="yes"
AC_MSG_CHECKING([if instrumentation-debug version was enabled])
AC_ARG_ENABLE([instrumentation-debug],
  AS_HELP_STRING([--enable-instrumentation-debug], [Enables generation of instrumentation-debug version]),
  [
    if test x$enableval = xyes -o x$enableval = x;
    then
        instrumentation_debug_disabled="no"
        AC_MSG_RESULT([yes])
        USED_VERSION="instrumentation-debug"
        VERSIONS+=" instrumentation-debug"
    else if test x$enableval = xno;
         then
            AC_MSG_RESULT([no])
            instrumentation_debug_disabled="yes"
         else
            AC_MSG_ERROR([This option can only be given 'yes' or 'no' values])
         fi
    fi
  ],
  [
      AC_MSG_RESULT([no])
      instrumentation_debug_disabled="yes"
  ]
)

AM_CONDITIONAL([INSTRUMENTATION_DEBUG_VERSION], [test x$instrumentation_debug_disabled = xno])
AC_SUBST([instrumentation_debugdir],['${libdir}/instrumentation-debug'])

# Enable instrumentation version
instrumentation_disabled="no"
AC_MSG_CHECKING([if instrumentation version was disabled])
AC_ARG_ENABLE([instrumentation],
  AS_HELP_STRING([--disable-instrumentation], [Disables generation of instrumentation version]),
  [
    if test x$disableval = xyes -o x$enableval = xno;
    then
        instrumentation_disabled="yes"
        AC_MSG_RESULT([yes])
    else if test x$disableval = xno -o x$disableval = x;
         then
            AC_MSG_RESULT([no])
            instrumentation_disabled="no"
            USED_VERSION="instrumentation"
            VERSIONS+=" instrumentation"
         else
            AC_MSG_ERROR([This option can only be given 'yes' or 'no' values])
         fi
    fi
  ],
  [
      AC_MSG_RESULT([no])
      instrumentation_disabled="no"
      USED_VERSION="instrumentation"
      VERSIONS+=" instrumentation"
  ]
)

AM_CONDITIONAL([INSTRUMENTATION_VERSION], [test x$instrumentation_disabled = xno])
AC_SUBST([instrumentationdir],['${libdir}/instrumentation'])

# Enable debug version
debug_disabled="no"
AC_MSG_CHECKING([if debug version was disabled])
AC_ARG_ENABLE([debug],
  AS_HELP_STRING([--disable-debug], [Disables generation of debug version]),
  [
    if test x$disableval = xyes -o x$enableval = xno;
    then
        debug_disabled="yes"
        AC_MSG_RESULT([yes])
    else if test x$disableval = xno -o x$disableval = x;
         then
            AC_MSG_RESULT([no])
            debug_disabled="no"
            USED_VERSION="debug"
            VERSIONS+=" debug"
         else
            AC_MSG_ERROR([This option can only be given 'yes' or 'no' values])
         fi
    fi
  ],
  [
      AC_MSG_RESULT([no])
      debug_disabled="no"
      USED_VERSION="debug"
      VERSIONS+=" debug"
  ]
)

AM_CONDITIONAL([DEBUG_VERSION], [test x$debug_disabled = xno])
AC_SUBST([debugdir],['${libdir}/debug'])

# Enable performance version
performance_disabled="no"
AC_MSG_CHECKING([if performance version was disabled])
AC_ARG_ENABLE([performance],
  AS_HELP_STRING([--disable-performance], [Disables generation of performance version]),
  [
    if test x$disableval = xyes -o x$enableval = xno;
    then
        performance_disabled="yes"
        AC_MSG_RESULT([yes])
    else if test x$disableval = xno -o x$disableval = x;
         then
            AC_MSG_RESULT([no])
            performance_disabled="no"
            USED_VERSION="performance"
            VERSIONS+=" performance"
         else
            AC_MSG_ERROR([This option can only be given 'yes' or 'no' values])
         fi
    fi
  ],
  [
      AC_MSG_RESULT([no])
      performance_disabled="no"
      USED_VERSION="performance"
      VERSIONS+=" performance"
  ]
)

AM_CONDITIONAL([PERFORMANCE_VERSION], [test x$performance_disabled = xno])
AC_SUBST([performancedir],['${libdir}/performance'])

AC_MSG_CHECKING([if any version is to be compiled])
if test x$performance_disabled = xyes -a x$debug_disabled = xyes -a x$instrumentation_disabled = xyes -a x$instrumentation_debug_disabled = xyes;
then
	AC_MSG_RESULT([no])
	AC_MSG_ERROR([At least one version needs to be compiled])
else
	AC_MSG_RESULT([yes])
fi
AC_SUBST([USED_VERSION])


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
AC_ARG_ENABLE([resiliency],
  AS_HELP_STRING([--enable-resiliency], [Enables task-level resiliency]),
  [
    AC_MSG_RESULT([$enableval])
    if test "x$enableval" = "xyes"; then
      AX_CHECK_COMPILE_FLAG([-fnon-call-exceptions],
        [],
        [AC_MSG_ERROR([resiliency mechanism depends on using compiler flag -fnon-call-exceptions])],
        [-Werror])
    fi
  ],
  [
    resiliency=no
    AC_MSG_RESULT([no])
  ]
)

AM_CONDITIONAL([RESILIENCY], [test "x$resiliency" = "xyes"])
AC_SUBST([resiliency])


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
echo ""
echo "Extra options:" $OPTIONS
