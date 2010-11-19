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

SUBDIRS = performance debug instrumentation

include $(top_srcdir)/src/common.am

#noinst_LTLIBRARIES = libcluster.la
#lib_LTLIBRARIES = libnanox-pe-cluster-mpi.la libnanox-pe-cluster-udp.la

EXTRA_DIST= \
	$(END)


#libcluster_la_CXXFLAGS=$(common_includes) $(common_cxxflags) $(ult_flags)
##libcluster_la_LDFLAGS = 
#libcluster_la_SOURCES = \
#		clusterdevice.hpp \
#		clusterdevice.cpp \
#		clusterdd.hpp \
#		clusterdd.cpp \
#		clusternode.hpp \
#		clusternode.cpp \
#		clusterthread.hpp \
#		clusterthread.cpp \
#		clusterinfo.hpp \
#		clusterinfo.cpp \
#		clusterutils.hpp \
#		$(END)
#
#
#
#libnanox_pe_cluster_mpi_la_CXXFLAGS=$(common_includes) $(common_cxxflags) $(ult_flags) -DGASNET_PAR -I@GASNET_DIR@/include -I@GASNET_DIR@/include/mpi-conduit
#libnanox_pe_cluster_mpi_la_LDFLAGS = $(ld_plugin_flags) -L@GASNET_DIR@/lib -lgasnet-mpi-par -lammpi -L/opt/osshpc/mpich-mx/32/lib/shared -L/opt/osshpc/mx/lib -lmpich -lmyriexpress -L/opt/ibmcmp/vacpp/10.1/lib -lxl -lpthread -L/gpfs/apps/GCC/4.4.0/lib/
#libnanox_pe_cluster_mpi_la_SOURCES = \
#		clusterplugin.cpp \
#		gasnetapi.hpp \
#		gasnetapi.cpp \
#		$(END)
#
#libnanox_pe_cluster_udp_la_CXXFLAGS=$(common_includes) $(common_cxxflags) $(ult_flags) -DGASNET_PAR -I@GASNET_DIR@/include -I@GASNET_DIR@/include/udp-conduit
#libnanox_pe_cluster_udp_la_LDFLAGS = $(ld_plugin_flags) -L@GASNET_DIR@/lib -lgasnet-udp-par -lamudp -L/gpfs/apps/GCC/4.4.0/lib/
#libnanox_pe_cluster_udp_la_SOURCES = \
#		clusterplugin.cpp \
#		gasnetapi.hpp \
#		gasnetapi.cpp \
#		$(END)
