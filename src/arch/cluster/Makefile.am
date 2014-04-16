#####################################################################################
#      Copyright 2009 Barcelona Supercomputing Center                               #
#                                                                                   #
#      This file is part of the NANOS++ library.                                    #
#                                                                                   #
#      NANOS++ is free software: you can redistribute it and/or modify              #
#      it under the terms of the GNU Lesser General Public License as published by  #
#      the Free Software Foundation, either version 3 of the License, or            #
#      (at your option) any later version.                                          #
#                                                                                   #
#      NANOS++ is distributed in the hope that it will be useful,                   #
#      but WITHOUT ANY WARRANTY; without even the implied warranty of               #
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                #
#      GNU Lesser General Public License for more details.                          #
#                                                                                   #
#      You should have received a copy of the GNU Lesser General Public License     #
#      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             #
#####################################################################################

include ../../common.am

cluster_sources = \
		clusterdevice_decl.hpp \
		clusterdevice.cpp \
		clusternode_fwd.hpp \
		clusternode_decl.hpp \
		clusternode.cpp \
		clusterthread_decl.hpp \
		clusterthread.cpp \
		remoteworkdescriptor_decl.hpp \
		remoteworkdescriptor_fwd.hpp \
		remoteworkdescriptor.cpp \
		$(END)

pe_cluster_sources = \
		clusterplugin.cpp \
		clusterplugin_decl.hpp \
		clusterplugin_fwd.hpp \
		gasnetapi_decl.hpp \
		gasnetapi.cpp \
		$(END)


pe_cluster_common_includes = -DGASNET_PAR -isystem @GASNET_DIR@/include
pe_cluster_common_ldflags = -L@GASNET_DIR@/lib

pe_cluster_mpi_includes = $(pe_cluster_common_includes) -I@GASNET_DIR@/include/mpi-conduit
pe_cluster_mpi_ldflags = $(pe_cluster_common_ldflags) -lgasnet-mpi-par -lammpi @MPI_CLDFLAGS@

pe_cluster_udp_includes = $(pe_cluster_common_includes) -I@GASNET_DIR@/include/udp-conduit
pe_cluster_udp_ldflags = $(pe_cluster_common_ldflags) -lgasnet-udp-par -lamudp

pe_cluster_smp_includes = $(pe_cluster_common_includes) -I@GASNET_DIR@/include/smp-conduit
pe_cluster_smp_ldflags = $(pe_cluster_common_ldflags) -lgasnet-smp-par

pe_cluster_ibv_includes = $(pe_cluster_common_includes) -I@GASNET_DIR@/include/ibv-conduit -DGASNET_CONDUIT_IBV
pe_cluster_ibv_ldflags = $(pe_cluster_common_ldflags) -lgasnet-ibv-par -libverbs @MPI_CLDFLAGS@

pe_cluster_dbg_common_includes = -DGASNET_PAR -isystem @GASNET_DBG_DIR@/include
pe_cluster_dbg_common_ldflags = -L@GASNET_DBG_DIR@/lib

pe_cluster_dbg_mpi_includes = $(pe_cluster_dbg_common_includes) -I@GASNET_DBG_DIR@/include/mpi-conduit
pe_cluster_dbg_mpi_ldflags = $(pe_cluster_dbg_common_ldflags) -lgasnet-mpi-par -lammpi @MPI_CLDFLAGS@

pe_cluster_dbg_udp_includes = $(pe_cluster_dbg_common_includes) -I@GASNET_DBG_DIR@/include/udp-conduit
pe_cluster_dbg_udp_ldflags = $(pe_cluster_dbg_common_ldflags) -lgasnet-udp-par -lamudp

pe_cluster_dbg_smp_includes = $(pe_cluster_dbg_common_includes) -I@GASNET_DBG_DIR@/include/smp-conduit
pe_cluster_dbg_smp_ldflags = $(pe_cluster_dbg_common_ldflags) -lgasnet-smp-par

pe_cluster_dbg_ibv_includes = $(pe_cluster_dbg_common_includes) -I@GASNET_DBG_DIR@/include/ibv-conduit -DGASNET_CONDUIT_IBV
pe_cluster_dbg_ibv_ldflags = $(pe_cluster_dbg_common_ldflags) -lgasnet-ibv-par -libverbs @MPI_CLDFLAGS@

noinst_LTLIBRARIES = 

if DEBUG_VERSION
noinst_LTLIBRARIES += debug/libcluster.la

#nanos_debug_cxxflags=$(common_includes) $(cluster_common_debug_cxxflags)
nanos_debug_cxxflags=$(common_includes) $(common_debug_cxxflags)

debug_libcluster_la_CXXFLAGS=$(nanos_debug_cxxflags)
debug_libcluster_la_SOURCES=$(cluster_sources)

debug_LTLIBRARIES =

if CONDUIT_MPI
debug_LTLIBRARIES += debug/libnanox-pe-cluster-mpi.la
debug_libnanox_pe_cluster_mpi_la_CXXFLAGS=$(nanos_debug_cxxflags) $(pe_cluster_dbg_mpi_includes)
debug_libnanox_pe_cluster_mpi_la_LDFLAGS=$(ld_plugin_flags) $(pe_cluster_dbg_mpi_ldflags)
debug_libnanox_pe_cluster_mpi_la_SOURCES=$(pe_cluster_sources)
endif

if CONDUIT_UDP
debug_LTLIBRARIES += debug/libnanox-pe-cluster-udp.la
debug_libnanox_pe_cluster_udp_la_CXXFLAGS=$(nanos_debug_cxxflags) $(pe_cluster_dbg_udp_includes)
debug_libnanox_pe_cluster_udp_la_LDFLAGS=$(ld_plugin_flags) $(pe_cluster_dbg_udp_ldflags)
debug_libnanox_pe_cluster_udp_la_SOURCES=$(pe_cluster_sources)
endif

if CONDUIT_SMP
debug_LTLIBRARIES += debug/libnanox-pe-cluster-smp.la
debug_libnanox_pe_cluster_smp_la_CXXFLAGS=$(nanos_debug_cxxflags) $(pe_cluster_dbg_smp_includes)
debug_libnanox_pe_cluster_smp_la_LDFLAGS=$(ld_plugin_flags) $(pe_cluster_dbg_smp_ldflags)
debug_libnanox_pe_cluster_smp_la_SOURCES=$(pe_cluster_sources)
endif

if CONDUIT_IBV
debug_LTLIBRARIES += debug/libnanox-pe-cluster-ibv.la
debug_libnanox_pe_cluster_ibv_la_CXXFLAGS=$(nanos_debug_cxxflags) $(pe_cluster_dbg_ibv_includes)
debug_libnanox_pe_cluster_ibv_la_LDFLAGS=$(ld_plugin_flags) $(pe_cluster_dbg_ibv_ldflags)
debug_libnanox_pe_cluster_ibv_la_SOURCES=$(pe_cluster_sources)
endif

endif

if INSTRUMENTATION_VERSION
noinst_LTLIBRARIES += instrumentation/libcluster.la

#nanos_instr_cxxflags=$(common_includes) $(cluster_common_instr_cxxflags)
nanos_instr_cxxflags=$(common_includes) $(common_instr_cxxflags)

instrumentation_libcluster_la_CXXFLAGS=$(nanos_instr_cxxflags)
instrumentation_libcluster_la_SOURCES=$(cluster_sources)

instrumentation_LTLIBRARIES =

if CONDUIT_MPI
instrumentation_LTLIBRARIES += instrumentation/libnanox-pe-cluster-mpi.la
instrumentation_libnanox_pe_cluster_mpi_la_CXXFLAGS=$(nanos_instr_cxxflags) $(pe_cluster_mpi_includes)
instrumentation_libnanox_pe_cluster_mpi_la_LDFLAGS=$(ld_plugin_flags) $(pe_cluster_mpi_ldflags)
instrumentation_libnanox_pe_cluster_mpi_la_SOURCES=$(pe_cluster_sources)
endif

if CONDUIT_UDP
instrumentation_LTLIBRARIES += instrumentation/libnanox-pe-cluster-udp.la
instrumentation_libnanox_pe_cluster_udp_la_CXXFLAGS=$(nanos_instr_cxxflags) $(pe_cluster_udp_includes)
instrumentation_libnanox_pe_cluster_udp_la_LDFLAGS=$(ld_plugin_flags) $(pe_cluster_udp_ldflags)
instrumentation_libnanox_pe_cluster_udp_la_SOURCES=$(pe_cluster_sources)
endif

if CONDUIT_SMP
instrumentation_LTLIBRARIES += instrumentation/libnanox-pe-cluster-smp.la
instrumentation_libnanox_pe_cluster_smp_la_CXXFLAGS=$(nanos_instr_cxxflags) $(pe_cluster_smp_includes)
instrumentation_libnanox_pe_cluster_smp_la_LDFLAGS=$(ld_plugin_flags) $(pe_cluster_smp_ldflags)
instrumentation_libnanox_pe_cluster_smp_la_SOURCES=$(pe_cluster_sources)
endif

if CONDUIT_IBV
instrumentation_LTLIBRARIES += instrumentation/libnanox-pe-cluster-ibv.la
instrumentation_libnanox_pe_cluster_ibv_la_CXXFLAGS=$(nanos_instr_cxxflags) $(pe_cluster_ibv_includes)
instrumentation_libnanox_pe_cluster_ibv_la_LDFLAGS=$(ld_plugin_flags) $(pe_cluster_ibv_ldflags)
instrumentation_libnanox_pe_cluster_ibv_la_SOURCES=$(pe_cluster_sources)
endif

endif


if INSTRUMENTATION_DEBUG_VERSION
noinst_LTLIBRARIES += instrumentation-debug/libcluster.la

#nanos_instrdebug_cxxflags=$(common_includes) $(cluster_common_instrdebug_cxxflags)
nanos_instrdebug_cxxflags=$(common_includes) $(common_instrdebug_cxxflags)

instrumentation_debug_libcluster_la_CXXFLAGS=$(nanos_instrdebug_cxxflags)
instrumentation_debug_libcluster_la_SOURCES=$(cluster_sources)

instrumentation_debug_LTLIBRARIES =

if CONDUIT_MPI
instrumentation_debug_LTLIBRARIES += instrumentation-debug/libnanox-pe-cluster-mpi.la
instrumentation_debug_libnanox_pe_cluster_mpi_la_CXXFLAGS=$(nanos_instrdebug_cxxflags) $(pe_cluster_dbg_mpi_includes)
instrumentation_debug_libnanox_pe_cluster_mpi_la_LDFLAGS=$(ld_plugin_flags) $(pe_cluster_dbg_mpi_ldflags)
instrumentation_debug_libnanox_pe_cluster_mpi_la_SOURCES=$(pe_cluster_sources)
endif

if CONDUIT_UDP
instrumentation_debug_LTLIBRARIES += instrumentation-debug/libnanox-pe-cluster-udp.la
instrumentation_debug_libnanox_pe_cluster_udp_la_CXXFLAGS=$(nanos_instrdebug_cxxflags) $(pe_cluster_dbg_udp_includes)
instrumentation_debug_libnanox_pe_cluster_udp_la_LDFLAGS=$(ld_plugin_flags) $(pe_cluster_dbg_udp_ldflags)
instrumentation_debug_libnanox_pe_cluster_udp_la_SOURCES=$(pe_cluster_sources)
endif

if CONDUIT_SMP
instrumentation_debug_LTLIBRARIES += instrumentation-debug/libnanox-pe-cluster-smp.la
instrumentation_debug_libnanox_pe_cluster_smp_la_CXXFLAGS=$(nanos_instrdebug_cxxflags) $(pe_cluster_dbg_smp_includes)
instrumentation_debug_libnanox_pe_cluster_smp_la_LDFLAGS=$(ld_plugin_flags) $(pe_cluster_dbg_smp_ldflags)
instrumentation_debug_libnanox_pe_cluster_smp_la_SOURCES=$(pe_cluster_sources)
endif

if CONDUIT_IBV
instrumentation_debug_LTLIBRARIES += instrumentation-debug/libnanox-pe-cluster-ibv.la
instrumentation_debug_libnanox_pe_cluster_ibv_la_CXXFLAGS=$(nanos_instrdebug_cxxflags) $(pe_cluster_dbg_ibv_includes)
instrumentation_debug_libnanox_pe_cluster_ibv_la_LDFLAGS=$(ld_plugin_flags) $(pe_cluster_dbg_ibv_ldflags)
instrumentation_debug_libnanox_pe_cluster_ibv_la_SOURCES=$(pe_cluster_sources)
endif

endif

if PERFORMANCE_VERSION
noinst_LTLIBRARIES += performance/libcluster.la

#nanos_perfo_cxxflags=$(common_includes) $(cluster_common_perfo_cxxflags)
nanos_perfo_cxxflags=$(common_includes) $(common_perfo_cxxflags)

performance_libcluster_la_CXXFLAGS=$(nanos_perfo_cxxflags)
performance_libcluster_la_SOURCES=$(cluster_sources)

performance_LTLIBRARIES =

if CONDUIT_MPI
performance_LTLIBRARIES += performance/libnanox-pe-cluster-mpi.la
performance_libnanox_pe_cluster_mpi_la_CXXFLAGS=$(nanos_perfo_cxxflags) $(pe_cluster_mpi_includes)
performance_libnanox_pe_cluster_mpi_la_LDFLAGS=$(ld_plugin_flags) $(pe_cluster_mpi_ldflags)
performance_libnanox_pe_cluster_mpi_la_SOURCES=$(pe_cluster_sources)
endif

if CONDUIT_UDP
performance_LTLIBRARIES += performance/libnanox-pe-cluster-udp.la
performance_libnanox_pe_cluster_udp_la_CXXFLAGS=$(nanos_perfo_cxxflags) $(pe_cluster_udp_includes)
performance_libnanox_pe_cluster_udp_la_LDFLAGS=$(ld_plugin_flags) $(pe_cluster_udp_ldflags)
performance_libnanox_pe_cluster_udp_la_SOURCES=$(pe_cluster_sources)
endif

if CONDUIT_SMP
performance_LTLIBRARIES += performance/libnanox-pe-cluster-smp.la
performance_libnanox_pe_cluster_smp_la_CXXFLAGS=$(nanos_perfo_cxxflags) $(pe_cluster_smp_includes)
performance_libnanox_pe_cluster_smp_la_LDFLAGS=$(ld_plugin_flags) $(pe_cluster_smp_ldflags)
performance_libnanox_pe_cluster_smp_la_SOURCES=$(pe_cluster_sources)
endif

if CONDUIT_IBV
performance_LTLIBRARIES += performance/libnanox-pe-cluster-ibv.la
performance_libnanox_pe_cluster_ibv_la_CXXFLAGS=$(nanos_perfo_cxxflags) $(pe_cluster_ibv_includes)
performance_libnanox_pe_cluster_ibv_la_LDFLAGS=$(ld_plugin_flags) $(pe_cluster_ibv_ldflags)
performance_libnanox_pe_cluster_ibv_la_SOURCES=$(pe_cluster_sources)
endif

endif
EXTRA_DIST= \
	$(END)
