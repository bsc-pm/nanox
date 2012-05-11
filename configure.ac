
AC_INIT([nanox], [0.7a], [alex.duran@bsc.es])
AC_CONFIG_SRCDIR([src/core/system.cpp])

# Environment
AC_CANONICAL_BUILD
AC_CANONICAL_SYSTEM
AC_CANONICAL_TARGET

# Automake initialization
AM_INIT_AUTOMAKE([-Wall -Werror dist-bzip2])
AC_CONFIG_MACRO_DIR([m4])

# Set compiler default flags
: ${CFLAGS=""}
: ${CXXFLAGS=""}

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
NANOS_CONFIG_LIBS_PERFO=
NANOS_CONFIG_LIBS_DEBUG=
NANOS_CONFIG_LIBS_INSTR=
AC_SUBST([NANOS_CONFIG_LIBDIRS])
AC_SUBST([NANOS_CONFIG_LIBS_PERFO])
AC_SUBST([NANOS_CONFIG_LIBS_DEBUG])
AC_SUBST([NANOS_CONFIG_LIBS_INSTR])
AC_SUBST([NANOS_CONFIG_LIBS_INSTRDEBUG])

# Check for architecture
ARCHITECTURES=
OS=

AC_SUBST([ARCHITECTURES])
AC_SUBST([OS])
AC_SUBST([SMP_ARCH])

#AC_MSG_CHECKING([for architecture])
smp_ult=yes
case $host in
  i?86-*-linux*)
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"
    SMP_ARCH=x86
    AM_CONDITIONAL([ARCH_IS_64_BITS], [test x$SMP_ARCH = xx86-64])
    ;;
  x86_64-*-linux*)
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"
    SMP_ARCH=x86-64
    AM_CONDITIONAL([ARCH_IS_64_BITS], [test x$SMP_ARCH = xx86-64])
    ;;
  ia64-*-linux*)
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"
    SMP_ARCH=ia64
    AM_CONDITIONAL([ARCH_IS_64_BITS], [test x$SMP_ARCH = xia64])
    ;;
  armv71-*-linux*)
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"
    SMP_ARCH=armv71
    ;;
  powerpc-*-linux* | powerpc64-*-linux*)
    OS=unix-os
    ARCHITECTURES="$ARCHITECTURES smp"

    case $CC in
      *-m32*)
         SMP_ARCH=ppc32
         ;;
      *-m64*)
         SMP_ARCH=ppc64
         ;;
      *)
         # GCC generates ppc32 code by default
         SMP_ARCH=ppc32
         ;;
    esac
    
    AM_CONDITIONAL([ARCH_IS_64_BITS], [test x$SMP_ARCH = xppc64])

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
if test x$cluster_enabled = xyes; then

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

    AC_SUBST([GASNET_DIR])
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
fi
AM_CONDITIONAL([CONDUIT_MPI], [test x$with_mpi_conduit = xyes])
AM_CONDITIONAL([CONDUIT_UDP], [test x$with_udp_conduit = xyes])
AM_CONDITIONAL([CONDUIT_SMP], [test x$with_smp_conduit = xyes])
AM_CONDITIONAL([CONDUIT_IBV], [test x$with_ibv_conduit = xyes])

# Check if it has nVidia GPUs / CUDA is installed
gpu_valid=no
gpu_enabled=yes
CUDA=""

AC_MSG_CHECKING([if the use of GPU devices was disabled])
AC_ARG_ENABLE([gpu-arch],
  AS_HELP_STRING([--disable-gpu-arch], [Disables the use of GPU devices]),
  [
    if test x"$enableval" = x"yes" -o x"$enableval" = x"";
    then
        gpu_enabled="yes"
        AC_MSG_RESULT([no])
    else if test x"$enableval" = x"no";
         then
            AC_MSG_RESULT([yes])
            gpu_enabled="no"
         else
            AC_MSG_ERROR([This option can only be given 'yes' or 'no' values])
         fi
    fi
  ],
  [
    gpu_enabled="yes"
    AC_MSG_RESULT([no])
  ]
)

AC_ARG_WITH([cuda],
AS_HELP_STRING([--with-cuda=dir], [Directory of CUDA installation]),
	[
		CUDADIR="${withval}"
	],
	[
		CUDADIR="/usr/local/cuda"
	])

if test x$gpu_enabled = xyes ; then
    if test -d "$CUDADIR" ;
    then
        AC_PATH_PROG([NVCC], [nvcc], [], [$PATH$PATH_SEPARATOR$CUDADIR/bin])
        if test x$NVCC = x ; then
            AC_MSG_WARN([nvcc not found. Disabling GPU support])
            gpu_valid=no
        else
            CUDA_INC="$CUDADIR/include"
            CUDA_LIB="$CUDADIR/lib"
            if test -d "$CUDADIR/lib64"; then
                CUDA_LIB="$CUDADIR/lib64"
            fi

            AC_SUBST([CUDA_INC])
            AC_SUBST([CUDA_LIB])

            cuda_version_h="$CUDA_INC/cuda.h"
            if test -e $cuda_version_h ;
            then
               cuda_version_string=`sed -ne '/^\s*#\s*define\s\+CUDA_VERSION .*$/p' $cuda_version_h`
               cuda_version=`echo $cuda_version_string | sed -e 's/#define CUDA_VERSION \([0-9]*\).*$/\1/'`
            else
               cuda_version=0
            fi

            gpu_valid=yes
        fi
    else
             AC_MSG_WARN([Directory "$CUDADIR" does not exist. Disabling GPU Support])
             gpu_valid=no
    fi
fi

if test x$gpu_valid = xyes; then
	CFLAGS="$CFLAGS -DGPU_DEV -DNANOS_CUDA_VERSION=$cuda_version"
	CXXFLAGS="$CXXFLAGS -DGPU_DEV -DNANOS_CUDA_VERSION=$cuda_version"
	ARCHITECTURES="$ARCHITECTURES gpu"
fi

AM_CONDITIONAL([GPU_SUPPORT],[test $gpu_valid = yes])

AC_SUBST([gpu_valid])

# Allocator support
allocator_disabled=no
AC_MSG_CHECKING([if Nanos++ Allocator has been disabled])
AC_ARG_ENABLE([allocator],
  AS_HELP_STRING([--disable-allocator], [Disables Allocator module]),
  [
    allocator_disabled="yes"
    AC_MSG_RESULT([yes])
  ],
  [
    allocator_disabled="no"
    AC_MSG_RESULT([no])
  ]
)
if test x$allocator_disabled = xyes; then
	CFLAGS="$CFLAGS -DNANOS_DISABLE_ALLOCATOR"
	CXXFLAGS="$CXXFLAGS -DNANOS_DISABLE_ALLOCATOR"
fi

AC_MSG_CHECKING([for architecture])

# Add architecture libraries
for arch in $ARCHITECTURES; do
	NANOS_CONFIG_LIBS_PERFO="$NANOS_CONFIG_LIBS_PERFO $NANOS_CONFIG_LIBS \$(abs_top_builddir)/src/arch/$arch/performance/lib$arch.la"
	NANOS_CONFIG_LIBS_DEBUG="$NANOS_CONFIG_LIBS_DEBUG $NANOS_CONFIG_LIBS \$(abs_top_builddir)/src/arch/$arch/debug/lib$arch.la"
	NANOS_CONFIG_LIBS_INSTR="$NANOS_CONFIG_LIBS_INSTR $NANOS_CONFIG_LIBS \$(abs_top_builddir)/src/arch/$arch/instrumentation/lib$arch.la"
	NANOS_CONFIG_LIBS_INSTRDEBUG="$NANOS_CONFIG_LIBS_INSTRDEBUG $NANOS_CONFIG_LIBS \$(abs_top_builddir)/src/arch/$arch/instrumentation-debug/lib$arch.la"
done
NANOS_CONFIG_LIBS_PERFO="$NANOS_CONFIG_LIBS_PERFO \$(abs_top_builddir)/src/arch/$OS/performance/libos.la \$(abs_top_builddir)/src/support/performance/libsupport.la"
NANOS_CONFIG_LIBS_DEBUG="$NANOS_CONFIG_LIBS_DEBUG \$(abs_top_builddir)/src/arch/$OS/debug/libos.la \$(abs_top_builddir)/src/support/debug/libsupport.la"
NANOS_CONFIG_LIBS_INSTR="$NANOS_CONFIG_LIBS_INSTR \$(abs_top_builddir)/src/arch/$OS/instrumentation/libos.la \$(abs_top_builddir)/src/support/instrumentation/libsupport.la"
NANOS_CONFIG_LIBS_INSTRDEBUG="$NANOS_CONFIG_LIBS_INSTRDEBUG \$(abs_top_builddir)/src/arch/$OS/instrumentation-debug/libos.la \$(abs_top_builddir)/src/support/instrumentation-debug/libsupport.la"

AM_CONDITIONAL([SMP_SUPPORTS_ULT],[test $smp_ult = yes])

AC_MSG_RESULT([$ARCHITECTURES/$OS])

USED_VERSION=""

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
AC_SUBST([instrdebugdir],['${libdir}/instrumentation-debug'])

# Enable instrumentation version
instrumentation_disabled="no"
AC_MSG_CHECKING([if instrumentation version was disabled])
AC_ARG_ENABLE([instrumentation],
  AS_HELP_STRING([--disable-instrumentation], [Disables generation of instrumentation version]),
  [
    if test x$disableval = xyes -o x$disableval = x;
    then
        instrumentation_disabled="yes"
        AC_MSG_RESULT([yes])
    else if test x$disableval = xno;
         then
            AC_MSG_RESULT([no])
            instrumentation_disabled="no"
            USED_VERSION="instrumentation"
         else
            AC_MSG_ERROR([This option can only be given 'yes' or 'no' values])
         fi
    fi
  ],
  [
      AC_MSG_RESULT([no])
      instrumentation_disabled="no"
      USED_VERSION="instrumentation"
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
    if test x$disableval = xyes -o x$disableval = x;
    then
        debug_disabled="yes"
        AC_MSG_RESULT([yes])
    else if test x$disableval = xno;
         then
            AC_MSG_RESULT([no])
            debug_disabled="no"
            USED_VERSION="debug"
         else
            AC_MSG_ERROR([This option can only be given 'yes' or 'no' values])
         fi
    fi
  ],
  [
      AC_MSG_RESULT([no])
      debug_disabled="no"
      USED_VERSION="debug"
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
    if test x$disableval = xyes -o x$disableval = x;
    then
        performance_disabled="yes"
        AC_MSG_RESULT([yes])
    else if test x$disableval = xno;
         then
            AC_MSG_RESULT([no])
            performance_disabled="no"
            USED_VERSION="performance"
         else
            AC_MSG_ERROR([This option can only be given 'yes' or 'no' values])
         fi
    fi
  ],
  [
      AC_MSG_RESULT([no])
      performance_disabled="no"
      USED_VERSION="performance"
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

AC_ARG_WITH([mpitrace],
           AS_HELP_STRING([--with-mpitrace=dir], [Directory of Extrae installation (deprecated)]),
           [
                extrae_version_h=$withval/include/extrae_version.h
                if test -e $extrae_version_h ;
                then
                   extrae_version_string=`sed -ne '/^\s*#\s*define\s\+EXTRAE_VERSION .*$/p' $extrae_version_h`
                   extrae_version=`echo $extrae_version_string | sed -e 's/#define EXTRAE_VERSION EXTRAE_VERSION_NUMBER(\([0-9]*\),\([0-9]*\),\([0-9]*\).*$/\1\2\3/'`
                   if test "$extrae_version" -ge 220;
                   then
                      MPITRACE_HOME="$withval"
                      MPITRACE_INC="$withval/include"
                      MPITRACE_LIB="$withval/lib"
                      MPITRACE_BIN="$withval/bin"
                      AC_MSG_RESULT([checking if Extrae library is compatible... yes])
                   else
                      AC_MSG_ERROR([checking if Extrae library is compatible... no (Extrae > 2.2.0 needed)])
                   fi
                else
                   AC_MSG_ERROR([checking if Extrae library is compatible... no (Extrae > 2.2.0 needed)])
                fi
           ]
)
AC_ARG_WITH([extrae],
           AS_HELP_STRING([--with-extrae=dir], [Directory of Extrae installation]),
           [
                extrae_version_h=$withval/include/extrae_version.h
                if test -e $extrae_version_h ;
                then
                   extrae_version_string=`sed -ne '/^\s*#\s*define\s\+EXTRAE_VERSION .*$/p' $extrae_version_h`
                   extrae_version=`echo $extrae_version_string | sed -e 's/#define EXTRAE_VERSION EXTRAE_VERSION_NUMBER(\([0-9]*\),\([0-9]*\),\([0-9]*\).*$/\1\2\3/'`
                   if test "$extrae_version" -ge 220;
                   then
                      MPITRACE_HOME="$withval"
                      MPITRACE_INC="$withval/include"
                      MPITRACE_LIB="$withval/lib"
                      MPITRACE_BIN="$withval/bin"
                      AC_MSG_RESULT([checking if Extrae library is compatible... yes])
                   else
                      AC_MSG_ERROR([checking if Extrae library is compatible... no (Extrae > 2.2.0 needed)])
                   fi
                else
                   AC_MSG_ERROR([checking if Extrae library is compatible... no (Extrae > 2.2.0 needed)])
                fi
           ]
)

AC_SUBST([MPITRACE_HOME])
AC_SUBST([MPITRACE_INC])
AC_SUBST([MPITRACE_LIB])
AC_SUBST([MPITRACE_BIN])


AM_CONDITIONAL([INSTRUMENTATION_EXTRAE], test x"$MPITRACE_HOME" != x)

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

# Output files
AC_CONFIG_HEADERS([config.h])

AC_CONFIG_FILES([Makefile
                 src/Makefile
                 src/core/Makefile
                 src/core/performance/Makefile
                 src/core/debug/Makefile
                 src/core/instrumentation/Makefile
                 src/core/instrumentation-debug/Makefile
		 src/support/Makefile
		 src/support/performance/Makefile
		 src/support/debug/Makefile
		 src/support/instrumentation/Makefile
		 src/support/instrumentation-debug/Makefile
		 src/plugins/Makefile
		 src/plugins/sched/Makefile
		 src/plugins/sched/performance/Makefile
		 src/plugins/sched/debug/Makefile
		 src/plugins/sched/instrumentation/Makefile
		 src/plugins/sched/instrumentation-debug/Makefile
		 src/plugins/throttle/Makefile
		 src/plugins/throttle/performance/Makefile
		 src/plugins/throttle/debug/Makefile
		 src/plugins/throttle/instrumentation/Makefile
		 src/plugins/throttle/instrumentation-debug/Makefile
		 src/plugins/barr/Makefile
		 src/plugins/barr/performance/Makefile
		 src/plugins/barr/debug/Makefile
		 src/plugins/barr/instrumentation/Makefile
		 src/plugins/barr/instrumentation-debug/Makefile
		 src/plugins/slicers/Makefile
		 src/plugins/slicers/performance/Makefile
		 src/plugins/slicers/debug/Makefile
		 src/plugins/slicers/instrumentation/Makefile
		 src/plugins/slicers/instrumentation-debug/Makefile
		 src/plugins/instrumentation/Makefile
		 src/plugins/instrumentation/performance/Makefile
		 src/plugins/instrumentation/debug/Makefile
		 src/plugins/instrumentation/instrumentation/Makefile
		 src/plugins/instrumentation/instrumentation-debug/Makefile
		 src/plugins/worksharing/Makefile
		 src/plugins/worksharing/performance/Makefile
		 src/plugins/worksharing/debug/Makefile
		 src/plugins/worksharing/instrumentation/Makefile
		 src/plugins/worksharing/instrumentation-debug/Makefile
		 src/apis/Makefile
		 src/apis/c/Makefile
		 src/apis/c/performance/Makefile
		 src/apis/c/debug/Makefile
		 src/apis/c/instrumentation/Makefile
		 src/apis/c/instrumentation-debug/Makefile
		 src/pms/Makefile
		 src/pms/openmp/Makefile
		 src/pms/openmp/performance/Makefile
		 src/pms/openmp/debug/Makefile
		 src/pms/openmp/instrumentation/Makefile
		 src/pms/openmp/instrumentation-debug/Makefile
		 src/pms/ss/Makefile
		 src/pms/ss/performance/Makefile
		 src/pms/ss/debug/Makefile
		 src/pms/ss/instrumentation/Makefile
		 src/pms/ss/instrumentation-debug/Makefile
		 src/pms/chapel/Makefile
		 src/pms/chapel/performance/Makefile
		 src/pms/chapel/debug/Makefile
		 src/pms/chapel/instrumentation/Makefile
		 src/pms/chapel/instrumentation-debug/Makefile
                 src/arch/Makefile
		 src/arch/unix-os/Makefile
		 src/arch/unix-os/performance/Makefile
		 src/arch/unix-os/debug/Makefile
		 src/arch/unix-os/instrumentation/Makefile
		 src/arch/unix-os/instrumentation-debug/Makefile
		 src/arch/smp/Makefile
		 src/arch/smp/performance/Makefile
		 src/arch/smp/debug/Makefile
		 src/arch/smp/instrumentation/Makefile
		 src/arch/smp/instrumentation-debug/Makefile
		 src/arch/fpga/Makefile
		 src/arch/fpga/performance/Makefile
		 src/arch/fpga/debug/Makefile
		 src/arch/fpga/instrumentation/Makefile
		 src/arch/fpga/instrumentation-debug/Makefile
		 src/arch/spu/Makefile
		 src/arch/spu/performance/Makefile
		 src/arch/spu/debug/Makefile
		 src/arch/spu/instrumentation/Makefile
		 src/arch/spu/instrumentation-debug/Makefile
		 src/arch/gpu/Makefile
		 src/arch/gpu/performance/Makefile
		 src/arch/gpu/debug/Makefile
		 src/arch/gpu/instrumentation/Makefile
		 src/arch/gpu/instrumentation-debug/Makefile
		 src/arch/cluster/Makefile
		 src/arch/cluster/performance/Makefile
		 src/arch/cluster/debug/Makefile
		 src/arch/cluster/instrumentation/Makefile
		 src/arch/cluster/instrumentation-debug/Makefile
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
echo "Host arch:" $SMP_ARCH
echo "Configured architectures:" $ARCHITECTURES
if test x$cluster_enabled = xyes ; then
echo "With cluster/gasnet conduits:" $CONDUITS
if test x$have_C_mpi = xyes ; then
echo "MPI_CLDFLAGS: " $MPI_CLDFLAGS
fi
fi
echo ""
