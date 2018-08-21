#####################################################################################
#      Copyright 2009-2018 Barcelona Supercomputing Center                          #
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
#      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            #
#####################################################################################

include $(top_srcdir)/src/common.am

devinclude_HEADERS = clustermpiplugin_fwd.hpp

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
		gasnetapi_fwd.hpp \
		gasnetapi.cpp \
		netwd.cpp \
		netwd_decl.hpp \
		$(END)

pe_clustermpi_sources = \
		clustermpiplugin.cpp \
		clustermpiplugin_decl.hpp \
		clustermpiplugin_fwd.hpp \
		gasnetapi_decl.hpp \
		gasnetapi_fwd.hpp \
		gasnetapi.cpp \
		netwd.cpp \
		netwd_decl.hpp \
		$(END)

noinst_LTLIBRARIES =

if is_debug_enabled
noinst_LTLIBRARIES += debug/libcluster.la

debug_libcluster_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libcluster_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libcluster_la_SOURCES=$(cluster_sources)

debug_LTLIBRARIES =

if gasnet_aries_available
debug_LTLIBRARIES += debug/libnanox-pe-cluster-aries.la

debug_libnanox_pe_cluster_aries_la_CPPFLAGS=$(common_debug_CPPFLAGS) @gasnet_aries_inc@
debug_libnanox_pe_cluster_aries_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_pe_cluster_aries_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @gasnetlib@
debug_libnanox_pe_cluster_aries_la_LIBADD=@gasnet_aries_libs@
debug_libnanox_pe_cluster_aries_la_SOURCES=$(pe_cluster_sources)
endif

if gasnet_mxm_available
debug_LTLIBRARIES += debug/libnanox-pe-cluster-mxm.la
debug_LTLIBRARIES += debug/libnanox-pe-clustermpi-mxm.la

debug_libnanox_pe_cluster_mxm_la_CPPFLAGS=$(common_debug_CPPFLAGS) @gasnet_mxm_inc@
debug_libnanox_pe_cluster_mxm_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_pe_cluster_mxm_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
debug_libnanox_pe_cluster_mxm_la_LIBADD=@gasnet_mxm_libs@
debug_libnanox_pe_cluster_mxm_la_SOURCES=$(pe_cluster_sources)

debug_libnanox_pe_clustermpi_mxm_la_CPPFLAGS=$(common_debug_CPPFLAGS) @gasnet_mxm_inc@
debug_libnanox_pe_clustermpi_mxm_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_pe_clustermpi_mxm_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
debug_libnanox_pe_clustermpi_mxm_la_LIBADD=@gasnet_mxm_libs@
debug_libnanox_pe_clustermpi_mxm_la_SOURCES=$(pe_clustermpi_sources)
endif

if gasnet_ibv_available
debug_LTLIBRARIES += debug/libnanox-pe-cluster-ibv.la
debug_LTLIBRARIES += debug/libnanox-pe-clustermpi-ibv.la

debug_libnanox_pe_cluster_ibv_la_CPPFLAGS=$(common_debug_CPPFLAGS) @gasnet_ibv_inc@
debug_libnanox_pe_cluster_ibv_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_pe_cluster_ibv_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
debug_libnanox_pe_cluster_ibv_la_LIBADD=@gasnet_ibv_libs@
debug_libnanox_pe_cluster_ibv_la_SOURCES=$(pe_cluster_sources)

debug_libnanox_pe_clustermpi_ibv_la_CPPFLAGS=$(common_debug_CPPFLAGS) @gasnet_ibv_inc@
debug_libnanox_pe_clustermpi_ibv_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_pe_clustermpi_ibv_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
debug_libnanox_pe_clustermpi_ibv_la_LIBADD=@gasnet_ibv_libs@
debug_libnanox_pe_clustermpi_ibv_la_SOURCES=$(pe_clustermpi_sources)
endif

if gasnet_mpi_available
debug_LTLIBRARIES += debug/libnanox-pe-cluster-mpi.la
debug_LTLIBRARIES += debug/libnanox-pe-clustermpi-mpi.la

debug_libnanox_pe_cluster_mpi_la_CPPFLAGS=$(common_debug_CPPFLAGS) @gasnet_mpi_inc@
debug_libnanox_pe_cluster_mpi_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_pe_cluster_mpi_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
debug_libnanox_pe_cluster_mpi_la_LIBADD=@gasnet_mpi_libs@
debug_libnanox_pe_cluster_mpi_la_SOURCES=$(pe_cluster_sources)

debug_libnanox_pe_clustermpi_mpi_la_CPPFLAGS=$(common_debug_CPPFLAGS) @gasnet_mpi_inc@
debug_libnanox_pe_clustermpi_mpi_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_pe_clustermpi_mpi_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
debug_libnanox_pe_clustermpi_mpi_la_LIBADD=@gasnet_mpi_libs@
debug_libnanox_pe_clustermpi_mpi_la_SOURCES=$(pe_clustermpi_sources)
endif

if gasnet_smp_available
debug_LTLIBRARIES += debug/libnanox-pe-cluster-smp.la

debug_libnanox_pe_cluster_smp_la_CPPFLAGS=$(common_debug_CPPFLAGS) @gasnet_smp_inc@
debug_libnanox_pe_cluster_smp_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_pe_cluster_smp_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @gasnetlib@
debug_libnanox_pe_cluster_smp_la_LIBADD=@gasnet_smp_libs@
debug_libnanox_pe_cluster_smp_la_SOURCES=$(pe_cluster_sources)
endif

if gasnet_udp_available
debug_LTLIBRARIES += debug/libnanox-pe-cluster-udp.la

debug_libnanox_pe_cluster_udp_la_CPPFLAGS=$(common_debug_CPPFLAGS) @gasnet_udp_inc@
debug_libnanox_pe_cluster_udp_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_pe_cluster_udp_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @gasnetlib@
debug_libnanox_pe_cluster_udp_la_LIBADD=@gasnet_udp_libs@
debug_libnanox_pe_cluster_udp_la_SOURCES=$(pe_cluster_sources)
endif

endif

if is_instrumentation_debug_enabled
noinst_LTLIBRARIES += instrumentation-debug/libcluster.la

instrumentation_debug_libcluster_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libcluster_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libcluster_la_SOURCES=$(cluster_sources)

instrumentation_debug_LTLIBRARIES =

if gasnet_aries_available
instrumentation_debug_LTLIBRARIES += instrumentation-debug/libnanox-pe-cluster-aries.la

instrumentation_debug_libnanox_pe_cluster_aries_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) @gasnet_aries_inc@
instrumentation_debug_libnanox_pe_cluster_aries_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_pe_cluster_aries_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @gasnetlib@
instrumentation_debug_libnanox_pe_cluster_aries_la_LIBADD=@gasnet_aries_libs@
instrumentation_debug_libnanox_pe_cluster_aries_la_SOURCES=$(pe_cluster_sources)
endif

if gasnet_mxm_available
instrumentation_debug_LTLIBRARIES += instrumentation-debug/libnanox-pe-cluster-mxm.la
instrumentation_debug_LTLIBRARIES += instrumentation-debug/libnanox-pe-clustermpi-mxm.la

instrumentation_debug_libnanox_pe_cluster_mxm_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) @gasnet_mxm_inc@
instrumentation_debug_libnanox_pe_cluster_mxm_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_pe_cluster_mxm_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
instrumentation_debug_libnanox_pe_cluster_mxm_la_LIBADD=@gasnet_mxm_libs@
instrumentation_debug_libnanox_pe_cluster_mxm_la_SOURCES=$(pe_cluster_sources)

instrumentation_debug_libnanox_pe_clustermpi_mxm_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) @gasnet_mxm_inc@
instrumentation_debug_libnanox_pe_clustermpi_mxm_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_pe_clustermpi_mxm_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
instrumentation_debug_libnanox_pe_clustermpi_mxm_la_LIBADD=@gasnet_mxm_libs@
instrumentation_debug_libnanox_pe_clustermpi_mxm_la_SOURCES=$(pe_clustermpi_sources)
endif

if gasnet_ibv_available
instrumentation_debug_LTLIBRARIES += instrumentation-debug/libnanox-pe-cluster-ibv.la
instrumentation_debug_LTLIBRARIES += instrumentation-debug/libnanox-pe-clustermpi-ibv.la

instrumentation_debug_libnanox_pe_cluster_ibv_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) @gasnet_ibv_inc@
instrumentation_debug_libnanox_pe_cluster_ibv_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_pe_cluster_ibv_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
instrumentation_debug_libnanox_pe_cluster_ibv_la_LIBADD=@gasnet_ibv_libs@
instrumentation_debug_libnanox_pe_cluster_ibv_la_SOURCES=$(pe_cluster_sources)

instrumentation_debug_libnanox_pe_clustermpi_ibv_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) @gasnet_ibv_inc@
instrumentation_debug_libnanox_pe_clustermpi_ibv_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_pe_clustermpi_ibv_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
instrumentation_debug_libnanox_pe_clustermpi_ibv_la_LIBADD=@gasnet_ibv_libs@
instrumentation_debug_libnanox_pe_clustermpi_ibv_la_SOURCES=$(pe_clustermpi_sources)
endif

if gasnet_mpi_available
instrumentation_debug_LTLIBRARIES += instrumentation-debug/libnanox-pe-cluster-mpi.la
instrumentation_debug_LTLIBRARIES += instrumentation-debug/libnanox-pe-clustermpi-mpi.la

instrumentation_debug_libnanox_pe_cluster_mpi_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) @gasnet_mpi_inc@
instrumentation_debug_libnanox_pe_cluster_mpi_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_pe_cluster_mpi_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
instrumentation_debug_libnanox_pe_cluster_mpi_la_LIBADD=@gasnet_mpi_libs@
instrumentation_debug_libnanox_pe_cluster_mpi_la_SOURCES=$(pe_cluster_sources)

instrumentation_debug_libnanox_pe_clustermpi_mpi_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) @gasnet_mpi_inc@
instrumentation_debug_libnanox_pe_clustermpi_mpi_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_pe_clustermpi_mpi_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
instrumentation_debug_libnanox_pe_clustermpi_mpi_la_LIBADD=@gasnet_mpi_libs@
instrumentation_debug_libnanox_pe_clustermpi_mpi_la_SOURCES=$(pe_clustermpi_sources)
endif

if gasnet_smp_available
instrumentation_debug_LTLIBRARIES += instrumentation-debug/libnanox-pe-cluster-smp.la

instrumentation_debug_libnanox_pe_cluster_smp_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) @gasnet_smp_inc@
instrumentation_debug_libnanox_pe_cluster_smp_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_pe_cluster_smp_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @gasnetlib@
instrumentation_debug_libnanox_pe_cluster_smp_la_LIBADD=@gasnet_smp_libs@
instrumentation_debug_libnanox_pe_cluster_smp_la_SOURCES=$(pe_cluster_sources)
endif

if gasnet_udp_available
instrumentation_debug_LTLIBRARIES += instrumentation-debug/libnanox-pe-cluster-udp.la

instrumentation_debug_libnanox_pe_cluster_udp_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) @gasnet_udp_inc@
instrumentation_debug_libnanox_pe_cluster_udp_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_pe_cluster_udp_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @gasnetlib@
instrumentation_debug_libnanox_pe_cluster_udp_la_LIBADD=@gasnet_udp_libs@
instrumentation_debug_libnanox_pe_cluster_udp_la_SOURCES=$(pe_cluster_sources)
endif

endif

if is_instrumentation_enabled
noinst_LTLIBRARIES += instrumentation/libcluster.la

instrumentation_libcluster_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libcluster_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libcluster_la_SOURCES=$(cluster_sources)

instrumentation_LTLIBRARIES =

if gasnet_aries_available
instrumentation_LTLIBRARIES += instrumentation/libnanox-pe-cluster-aries.la

instrumentation_libnanox_pe_cluster_aries_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) @gasnet_aries_inc@
instrumentation_libnanox_pe_cluster_aries_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_pe_cluster_aries_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @gasnetlib@
instrumentation_libnanox_pe_cluster_aries_la_LIBADD=@gasnet_aries_libs@
instrumentation_libnanox_pe_cluster_aries_la_SOURCES=$(pe_cluster_sources)
endif

if gasnet_mxm_available
instrumentation_LTLIBRARIES += instrumentation/libnanox-pe-cluster-mxm.la
instrumentation_LTLIBRARIES += instrumentation/libnanox-pe-clustermpi-mxm.la

instrumentation_libnanox_pe_cluster_mxm_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) @gasnet_mxm_inc@
instrumentation_libnanox_pe_cluster_mxm_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_pe_cluster_mxm_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
instrumentation_libnanox_pe_cluster_mxm_la_LIBADD=@gasnet_mxm_libs@
instrumentation_libnanox_pe_cluster_mxm_la_SOURCES=$(pe_cluster_sources)

instrumentation_libnanox_pe_clustermpi_mxm_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) @gasnet_mxm_inc@
instrumentation_libnanox_pe_clustermpi_mxm_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_pe_clustermpi_mxm_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
instrumentation_libnanox_pe_clustermpi_mxm_la_LIBADD=@gasnet_mxm_libs@
instrumentation_libnanox_pe_clustermpi_mxm_la_SOURCES=$(pe_clustermpi_sources)
endif

if gasnet_ibv_available
instrumentation_LTLIBRARIES += instrumentation/libnanox-pe-cluster-ibv.la
instrumentation_LTLIBRARIES += instrumentation/libnanox-pe-clustermpi-ibv.la

instrumentation_libnanox_pe_cluster_ibv_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) @gasnet_ibv_inc@
instrumentation_libnanox_pe_cluster_ibv_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_pe_cluster_ibv_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
instrumentation_libnanox_pe_cluster_ibv_la_LIBADD=@gasnet_ibv_libs@
instrumentation_libnanox_pe_cluster_ibv_la_SOURCES=$(pe_cluster_sources)

instrumentation_libnanox_pe_clustermpi_ibv_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) @gasnet_ibv_inc@
instrumentation_libnanox_pe_clustermpi_ibv_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_pe_clustermpi_ibv_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
instrumentation_libnanox_pe_clustermpi_ibv_la_LIBADD=@gasnet_ibv_libs@
instrumentation_libnanox_pe_clustermpi_ibv_la_SOURCES=$(pe_clustermpi_sources)
endif

if gasnet_mpi_available
instrumentation_LTLIBRARIES += instrumentation/libnanox-pe-cluster-mpi.la
instrumentation_LTLIBRARIES += instrumentation/libnanox-pe-clustermpi-mpi.la

instrumentation_libnanox_pe_cluster_mpi_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) @gasnet_mpi_inc@
instrumentation_libnanox_pe_cluster_mpi_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_pe_cluster_mpi_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
instrumentation_libnanox_pe_cluster_mpi_la_LIBADD=@gasnet_mpi_libs@
instrumentation_libnanox_pe_cluster_mpi_la_SOURCES=$(pe_cluster_sources)

instrumentation_libnanox_pe_clustermpi_mpi_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) @gasnet_mpi_inc@
instrumentation_libnanox_pe_clustermpi_mpi_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_pe_clustermpi_mpi_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
instrumentation_libnanox_pe_clustermpi_mpi_la_LIBADD=@gasnet_mpi_libs@
instrumentation_libnanox_pe_clustermpi_mpi_la_SOURCES=$(pe_clustermpi_sources)
endif

if gasnet_smp_available
instrumentation_LTLIBRARIES += instrumentation/libnanox-pe-cluster-smp.la

instrumentation_libnanox_pe_cluster_smp_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) @gasnet_smp_inc@
instrumentation_libnanox_pe_cluster_smp_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_pe_cluster_smp_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @gasnetlib@
instrumentation_libnanox_pe_cluster_smp_la_LIBADD=@gasnet_smp_libs@
instrumentation_libnanox_pe_cluster_smp_la_SOURCES=$(pe_cluster_sources)
endif

if gasnet_udp_available
instrumentation_LTLIBRARIES += instrumentation/libnanox-pe-cluster-udp.la

instrumentation_libnanox_pe_cluster_udp_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) @gasnet_udp_inc@
instrumentation_libnanox_pe_cluster_udp_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_pe_cluster_udp_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @gasnetlib@
instrumentation_libnanox_pe_cluster_udp_la_LIBADD=@gasnet_udp_libs@
instrumentation_libnanox_pe_cluster_udp_la_SOURCES=$(pe_cluster_sources)
endif

endif

if is_performance_enabled
noinst_LTLIBRARIES += performance/libcluster.la

performance_libcluster_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libcluster_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libcluster_la_SOURCES=$(cluster_sources)

performance_LTLIBRARIES =

if gasnet_aries_available
performance_LTLIBRARIES += performance/libnanox-pe-cluster-aries.la

performance_libnanox_pe_cluster_aries_la_CPPFLAGS=$(common_performance_CPPFLAGS) @gasnet_aries_inc@
performance_libnanox_pe_cluster_aries_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_pe_cluster_aries_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @gasnetlib@
performance_libnanox_pe_cluster_aries_la_LIBADD=@gasnet_aries_libs@
performance_libnanox_pe_cluster_aries_la_SOURCES=$(pe_cluster_sources)
endif

if gasnet_mxm_available
performance_LTLIBRARIES += performance/libnanox-pe-cluster-mxm.la
performance_LTLIBRARIES += performance/libnanox-pe-clustermpi-mxm.la

performance_libnanox_pe_cluster_mxm_la_CPPFLAGS=$(common_performance_CPPFLAGS) @gasnet_mxm_inc@
performance_libnanox_pe_cluster_mxm_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_pe_cluster_mxm_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
performance_libnanox_pe_cluster_mxm_la_LIBADD=@gasnet_mxm_libs@
performance_libnanox_pe_cluster_mxm_la_SOURCES=$(pe_cluster_sources)

performance_libnanox_pe_clustermpi_mxm_la_CPPFLAGS=$(common_performance_CPPFLAGS) @gasnet_mxm_inc@
performance_libnanox_pe_clustermpi_mxm_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_pe_clustermpi_mxm_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
performance_libnanox_pe_clustermpi_mxm_la_LIBADD=@gasnet_mxm_libs@
performance_libnanox_pe_clustermpi_mxm_la_SOURCES=$(pe_clustermpi_sources)
endif

if gasnet_ibv_available
performance_LTLIBRARIES += performance/libnanox-pe-cluster-ibv.la
performance_LTLIBRARIES += performance/libnanox-pe-clustermpi-ibv.la

performance_libnanox_pe_cluster_ibv_la_CPPFLAGS=$(common_performance_CPPFLAGS) @gasnet_ibv_inc@
performance_libnanox_pe_cluster_ibv_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_pe_cluster_ibv_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
performance_libnanox_pe_cluster_ibv_la_LIBADD=@gasnet_ibv_libs@
performance_libnanox_pe_cluster_ibv_la_SOURCES=$(pe_cluster_sources)

performance_libnanox_pe_clustermpi_ibv_la_CPPFLAGS=$(common_performance_CPPFLAGS) @gasnet_ibv_inc@
performance_libnanox_pe_clustermpi_ibv_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_pe_clustermpi_ibv_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
performance_libnanox_pe_clustermpi_ibv_la_LIBADD=@gasnet_ibv_libs@
performance_libnanox_pe_clustermpi_ibv_la_SOURCES=$(pe_clustermpi_sources)
endif

if gasnet_mpi_available
performance_LTLIBRARIES += performance/libnanox-pe-cluster-mpi.la
performance_LTLIBRARIES += performance/libnanox-pe-clustermpi-mpi.la

performance_libnanox_pe_cluster_mpi_la_CPPFLAGS=$(common_performance_CPPFLAGS) @gasnet_mpi_inc@
performance_libnanox_pe_cluster_mpi_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_pe_cluster_mpi_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
performance_libnanox_pe_cluster_mpi_la_LIBADD=@gasnet_mpi_libs@
performance_libnanox_pe_cluster_mpi_la_SOURCES=$(pe_cluster_sources)

performance_libnanox_pe_clustermpi_mpi_la_CPPFLAGS=$(common_performance_CPPFLAGS) @gasnet_mpi_inc@
performance_libnanox_pe_clustermpi_mpi_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_pe_clustermpi_mpi_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @mpilib@ @gasnetlib@
performance_libnanox_pe_clustermpi_mpi_la_LIBADD=@gasnet_mpi_libs@
performance_libnanox_pe_clustermpi_mpi_la_SOURCES=$(pe_clustermpi_sources)
endif

if gasnet_smp_available
performance_LTLIBRARIES += performance/libnanox-pe-cluster-smp.la

performance_libnanox_pe_cluster_smp_la_CPPFLAGS=$(common_performance_CPPFLAGS) @gasnet_smp_inc@
performance_libnanox_pe_cluster_smp_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_pe_cluster_smp_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @gasnetlib@
performance_libnanox_pe_cluster_smp_la_LIBADD=@gasnet_smp_libs@
performance_libnanox_pe_cluster_smp_la_SOURCES=$(pe_cluster_sources)
endif

if gasnet_udp_available
performance_LTLIBRARIES += performance/libnanox-pe-cluster-udp.la

performance_libnanox_pe_cluster_udp_la_CPPFLAGS=$(common_performance_CPPFLAGS) @gasnet_udp_inc@
performance_libnanox_pe_cluster_udp_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_pe_cluster_udp_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @gasnetlib@
performance_libnanox_pe_cluster_udp_la_LIBADD=@gasnet_udp_libs@
performance_libnanox_pe_cluster_udp_la_SOURCES=$(pe_cluster_sources)
endif

endif

EXTRA_DIST= \
	$(END)
