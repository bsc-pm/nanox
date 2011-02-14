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

include $(top_srcdir)/src/arch/cluster/version_commons.am
include $(top_srcdir)/src/common.am

if INSTRUMENTATION_DEBUG_VERSION
noinst_LTLIBRARIES = libcluster.la

nanos_instrdebug_cxxflags=$(common_includes) $(common_instrdebug_cxxflags)

libcluster_la_CXXFLAGS=$(nanos_instrdebug_cxxflags)
libcluster_la_SOURCES=$(cluster_sources)

instrdebug_LTLIBRARIES =

if CONDUIT_MPI
instrdebug_LTLIBRARIES += libnanox-pe-cluster-mpi.la
libnanox_pe_cluster_mpi_la_CXXFLAGS=$(nanos_instrdebug_cxxflags) $(pe_cluster_mpi_includes)
libnanox_pe_cluster_mpi_la_LDFLAGS=$(ld_plugin_flags) $(pe_cluster_mpi_ldflags)
libnanox_pe_cluster_mpi_la_SOURCES=$(pe_cluster_sources)
endif

if CONDUIT_UDP
instrdebug_LTLIBRARIES += libnanox-pe-cluster-udp.la
libnanox_pe_cluster_udp_la_CXXFLAGS=$(nanos_instrdebug_cxxflags) $(pe_cluster_udp_includes)
libnanox_pe_cluster_udp_la_LDFLAGS=$(ld_plugin_flags) $(pe_cluster_udp_ldflags)
libnanox_pe_cluster_udp_la_SOURCES=$(pe_cluster_sources)
endif

if CONDUIT_SMP
instrdebug_LTLIBRARIES += libnanox-pe-cluster-smp.la
libnanox_pe_cluster_smp_la_CXXFLAGS=$(nanos_instrdebug_cxxflags) $(pe_cluster_smp_includes)
libnanox_pe_cluster_smp_la_LDFLAGS=$(ld_plugin_flags) $(pe_cluster_smp_ldflags)
libnanox_pe_cluster_smp_la_SOURCES=$(pe_cluster_sources)
endif

endif
