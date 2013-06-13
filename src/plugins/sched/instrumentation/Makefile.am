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

include $(top_srcdir)/src/plugins/sched/version_commons.am
include $(top_srcdir)/src/common.am

if INSTRUMENTATION_VERSION
instrumentation_LTLIBRARIES = \
	libnanox-sched-bf.la \
	libnanox-sched-dbf.la \
 	libnanox-sched-wf.la \
 	libnanox-sched-affinity.la \
 	libnanox-sched-versioning.la \
 	libnanox-sched-affinity-smartpriority.la \
 	libnanox-sched-socket.la \
	$(END)

nanos_instr_cxxflags=$(common_includes) $(common_instr_cxxflags)
if IBM_C
nanos_instr_cxxflags += -qnoinline
endif

libnanox_sched_bf_la_CXXFLAGS = $(nanos_instr_cxxflags)
libnanox_sched_bf_la_LDFLAGS = $(ld_plugin_flags)
libnanox_sched_bf_la_SOURCES = $(bf_sources)

libnanox_sched_dbf_la_CXXFLAGS = $(nanos_instr_cxxflags)
libnanox_sched_dbf_la_LDFLAGS = $(ld_plugin_flags)
libnanox_sched_dbf_la_SOURCES = $(dbf_sources)

libnanox_sched_wf_la_CXXFLAGS = $(nanos_instr_cxxflags)
libnanox_sched_wf_la_LDFLAGS = $(ld_plugin_flags)
libnanox_sched_wf_la_SOURCES = $(wf_sources)
 
libnanox_sched_affinity_la_CXXFLAGS = $(nanos_instr_cxxflags)
libnanox_sched_affinity_la_LDFLAGS = $(ld_plugin_flags)
libnanox_sched_affinity_la_SOURCES = $(affinity_sources)

libnanox_sched_versioning_la_CXXFLAGS = $(nanos_instr_cxxflags)
libnanox_sched_versioning_la_LDFLAGS = $(ld_plugin_flags)
libnanox_sched_versioning_la_SOURCES = $(versioning_sources)
 
libnanox_sched_affinity_smartpriority_la_CXXFLAGS = $(nanos_instr_cxxflags)
libnanox_sched_affinity_smartpriority_la_LDFLAGS = $(ld_plugin_flags)
libnanox_sched_affinity_smartpriority_la_SOURCES = $(affinity_smartpriority_sources)
 
if HWLOC
libnanox_sched_socket_la_CXXFLAGS = $(nanos_instr_cxxflags) -DHWLOC -I@HWLOC_INC@
libnanox_sched_socket_la_LDFLAGS = $(ld_plugin_flags) -L@HWLOC_LIB@ -lhwloc
else
libnanox_sched_socket_la_CXXFLAGS = $(nanos_instr_cxxflags)
libnanox_sched_socket_la_LDFLAGS = $(ld_plugin_flags)
endif
libnanox_sched_socket_la_SOURCES = $(socket_sources)
endif

