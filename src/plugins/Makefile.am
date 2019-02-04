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

debug_LTLIBRARIES =
instrumentation_LTLIBRARIES =
instrumentation_debug_LTLIBRARIES =
performance_LTLIBRARIES =

devinclude_HEADERS=\
	instrumentation/ompt-headers/ompt.h \
	instrumentation/ompt-headers/ompt-api.h \
	instrumentation/ompt-headers/ompt-events.h \
	instrumentation/ompt-headers/ompt-initialize.h \
	instrumentation/ompt-headers/ompt-records.h \
	instrumentation/ompt-headers/ompt-states.h \
	instrumentation/ompt-headers/ompt-types.h \
	instrumentation/ompt-headers/ompt-types-callbacks.h \
	$(END)




######################################################################################################
#### Schedule ########################################################################################
######################################################################################################
bf_sources=\
	sched/bf_sched.cpp \
	$(END)

mpq_sources=\
	sched/mpq_sched.cpp \
	$(END)

dbf_sources=\
	sched/dbf_sched.cpp \
	$(END)

wf_sources=\
	sched/wf_sched.cpp \
	$(END)

affinity_sources=\
	sched/affinity_sched.cpp \
	$(END)

affinity_ready_sources=\
	sched/affinity-ready_sched.cpp \
	$(END)

versioning_sources=\
	sched/versioning_sched.cpp \
	$(END)
	
socket_sources=\
	sched/socket_sched.cpp \
	$(END)

botlev_sources=\
	sched/botlev_sched.cpp \
	$(END)

if is_debug_enabled
debug_LTLIBRARIES +=\
 debug/libnanox-sched-bf.la\
 debug/libnanox-sched-mpq.la\
 debug/libnanox-sched-dbf.la\
 debug/libnanox-sched-wf.la\
 debug/libnanox-sched-affinity.la\
 debug/libnanox-sched-affinity-ready.la\
 debug/libnanox-sched-versioning.la\
 debug/libnanox-sched-socket.la\
 debug/libnanox-sched-botlev.la

debug_libnanox_sched_bf_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_sched_bf_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_sched_bf_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_sched_bf_la_SOURCES=$(bf_sources)

debug_libnanox_sched_mpq_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_sched_mpq_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_sched_mpq_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_sched_mpq_la_SOURCES=$(mpq_sources)

debug_libnanox_sched_dbf_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_sched_dbf_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_sched_dbf_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_sched_dbf_la_SOURCES=$(dbf_sources)

debug_libnanox_sched_wf_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_sched_wf_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_sched_wf_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_sched_wf_la_SOURCES=$(wf_sources)

debug_libnanox_sched_affinity_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_sched_affinity_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_sched_affinity_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_sched_affinity_la_SOURCES=$(affinity_sources)

debug_libnanox_sched_affinity_ready_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_sched_affinity_ready_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_sched_affinity_ready_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_sched_affinity_ready_la_SOURCES=$(affinity_ready_sources)

debug_libnanox_sched_versioning_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_sched_versioning_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_sched_versioning_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_sched_versioning_la_SOURCES=$(versioning_sources)

debug_libnanox_sched_socket_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_sched_socket_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_sched_socket_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_sched_socket_la_SOURCES=$(socket_sources)

debug_libnanox_sched_botlev_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_sched_botlev_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_sched_botlev_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_sched_botlev_la_SOURCES=$(botlev_sources)
endif

if is_instrumentation_debug_enabled
instrumentation_debug_LTLIBRARIES +=\
 instrumentation-debug/libnanox-sched-bf.la\
 instrumentation-debug/libnanox-sched-mpq.la\
 instrumentation-debug/libnanox-sched-dbf.la\
 instrumentation-debug/libnanox-sched-wf.la\
 instrumentation-debug/libnanox-sched-affinity.la\
 instrumentation-debug/libnanox-sched-affinity-ready.la\
 instrumentation-debug/libnanox-sched-versioning.la\
 instrumentation-debug/libnanox-sched-socket.la\
 instrumentation-debug/libnanox-sched-botlev.la

instrumentation_debug_libnanox_sched_bf_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_sched_bf_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_sched_bf_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_sched_bf_la_SOURCES=$(bf_sources)

instrumentation_debug_libnanox_sched_mpq_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_sched_mpq_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_sched_mpq_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_sched_mpq_la_SOURCES=$(mpq_sources)

instrumentation_debug_libnanox_sched_dbf_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_sched_dbf_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_sched_dbf_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_sched_dbf_la_SOURCES=$(dbf_sources)

instrumentation_debug_libnanox_sched_wf_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_sched_wf_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_sched_wf_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_sched_wf_la_SOURCES=$(wf_sources)

instrumentation_debug_libnanox_sched_affinity_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_sched_affinity_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_sched_affinity_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_sched_affinity_la_SOURCES=$(affinity_sources)

instrumentation_debug_libnanox_sched_affinity_ready_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_sched_affinity_ready_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_sched_affinity_ready_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_sched_affinity_ready_la_SOURCES=$(affinity_ready_sources)

instrumentation_debug_libnanox_sched_versioning_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_sched_versioning_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_sched_versioning_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_sched_versioning_la_SOURCES=$(versioning_sources)

instrumentation_debug_libnanox_sched_socket_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_sched_socket_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_sched_socket_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_sched_socket_la_SOURCES=$(socket_sources)

instrumentation_debug_libnanox_sched_botlev_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_sched_botlev_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_sched_botlev_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_sched_botlev_la_SOURCES=$(botlev_sources)
endif

if is_instrumentation_enabled
instrumentation_LTLIBRARIES +=\
 instrumentation/libnanox-sched-bf.la\
 instrumentation/libnanox-sched-mpq.la\
 instrumentation/libnanox-sched-dbf.la\
 instrumentation/libnanox-sched-wf.la\
 instrumentation/libnanox-sched-affinity.la\
 instrumentation/libnanox-sched-affinity-ready.la\
 instrumentation/libnanox-sched-versioning.la\
 instrumentation/libnanox-sched-socket.la\
 instrumentation/libnanox-sched-botlev.la

instrumentation_libnanox_sched_bf_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_sched_bf_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_sched_bf_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_sched_bf_la_SOURCES=$(bf_sources)

instrumentation_libnanox_sched_mpq_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_sched_mpq_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_sched_mpq_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_sched_mpq_la_SOURCES=$(mpq_sources)

instrumentation_libnanox_sched_dbf_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_sched_dbf_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_sched_dbf_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_sched_dbf_la_SOURCES=$(dbf_sources)

instrumentation_libnanox_sched_wf_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_sched_wf_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_sched_wf_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_sched_wf_la_SOURCES=$(wf_sources)

instrumentation_libnanox_sched_affinity_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_sched_affinity_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_sched_affinity_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_sched_affinity_la_SOURCES=$(affinity_sources)

instrumentation_libnanox_sched_affinity_ready_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_sched_affinity_ready_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_sched_affinity_ready_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_sched_affinity_ready_la_SOURCES=$(affinity_ready_sources)

instrumentation_libnanox_sched_versioning_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_sched_versioning_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_sched_versioning_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_sched_versioning_la_SOURCES=$(versioning_sources)

instrumentation_libnanox_sched_socket_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_sched_socket_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_sched_socket_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_sched_socket_la_SOURCES=$(socket_sources)

instrumentation_libnanox_sched_botlev_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_sched_botlev_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_sched_botlev_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_sched_botlev_la_SOURCES=$(botlev_sources)
endif

if is_performance_enabled
performance_LTLIBRARIES +=\
 performance/libnanox-sched-bf.la\
 performance/libnanox-sched-mpq.la\
 performance/libnanox-sched-dbf.la\
 performance/libnanox-sched-wf.la\
 performance/libnanox-sched-affinity.la\
 performance/libnanox-sched-affinity-ready.la\
 performance/libnanox-sched-versioning.la\
 performance/libnanox-sched-socket.la\
 performance/libnanox-sched-botlev.la

performance_libnanox_sched_bf_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_sched_bf_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_sched_bf_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_sched_bf_la_SOURCES=$(bf_sources)

performance_libnanox_sched_mpq_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_sched_mpq_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_sched_mpq_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_sched_mpq_la_SOURCES=$(mpq_sources)

performance_libnanox_sched_dbf_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_sched_dbf_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_sched_dbf_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_sched_dbf_la_SOURCES=$(dbf_sources)

performance_libnanox_sched_wf_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_sched_wf_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_sched_wf_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_sched_wf_la_SOURCES=$(wf_sources)

performance_libnanox_sched_affinity_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_sched_affinity_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_sched_affinity_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_sched_affinity_la_SOURCES=$(affinity_sources)

performance_libnanox_sched_affinity_ready_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_sched_affinity_ready_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_sched_affinity_ready_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_sched_affinity_ready_la_SOURCES=$(affinity_ready_sources)

performance_libnanox_sched_versioning_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_sched_versioning_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_sched_versioning_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_sched_versioning_la_SOURCES=$(versioning_sources)

performance_libnanox_sched_socket_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_sched_socket_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_sched_socket_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_sched_socket_la_SOURCES=$(socket_sources)

performance_libnanox_sched_botlev_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_sched_botlev_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_sched_botlev_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_sched_botlev_la_SOURCES=$(botlev_sources)
endif

######################################################################################################
######################################################################################################


######################################################################################################
#### Throttle ########################################################################################
######################################################################################################
hysteresis_sources=\
	throttle/hysteresis_throttle.cpp \
	$(END)

dummy_sources=\
	throttle/dummy_throttle.cpp \
	$(END)

numtasks_sources=\
	throttle/numtasks_throttle.cpp \
	$(END)

idlethreads_sources=\
	throttle/idlethreads_throttle.cpp \
	$(END)

taskdepth_sources=\
	throttle/taskdepth_throttle.cpp \
	$(END)

readytasks_sources=\
	throttle/readytasks_throttle.cpp \
	$(END)


if is_debug_enabled
debug_LTLIBRARIES += \
	debug/libnanox-throttle-hysteresis.la \
	debug/libnanox-throttle-dummy.la \
	debug/libnanox-throttle-numtasks.la \
	debug/libnanox-throttle-idlethreads.la \
	debug/libnanox-throttle-taskdepth.la \
	debug/libnanox-throttle-readytasks.la \
	$(END)

debug_libnanox_throttle_hysteresis_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_throttle_hysteresis_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_throttle_hysteresis_la_SOURCES=$(hysteresis_sources)

debug_libnanox_throttle_dummy_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_throttle_dummy_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_throttle_dummy_la_SOURCES=$(dummy_sources)

debug_libnanox_throttle_numtasks_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_throttle_numtasks_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_throttle_numtasks_la_SOURCES=$(numtasks_sources)

debug_libnanox_throttle_idlethreads_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_throttle_idlethreads_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_throttle_idlethreads_la_SOURCES=$(idlethreads_sources)

debug_libnanox_throttle_taskdepth_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_throttle_taskdepth_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_throttle_taskdepth_la_SOURCES=$(taskdepth_sources)

debug_libnanox_throttle_readytasks_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_throttle_readytasks_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_throttle_readytasks_la_SOURCES=$(readytasks_sources)
endif

if is_instrumentation_enabled
instrumentation_LTLIBRARIES += \
	instrumentation/libnanox-throttle-hysteresis.la \
	instrumentation/libnanox-throttle-dummy.la \
	instrumentation/libnanox-throttle-numtasks.la \
	instrumentation/libnanox-throttle-idlethreads.la \
	instrumentation/libnanox-throttle-taskdepth.la \
	instrumentation/libnanox-throttle-readytasks.la \
	$(END)

instrumentation_libnanox_throttle_hysteresis_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_throttle_hysteresis_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_throttle_hysteresis_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_throttle_hysteresis_la_SOURCES=$(hysteresis_sources)

instrumentation_libnanox_throttle_dummy_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_throttle_dummy_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_throttle_dummy_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_throttle_dummy_la_SOURCES=$(dummy_sources)

instrumentation_libnanox_throttle_numtasks_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_throttle_numtasks_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_throttle_numtasks_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_throttle_numtasks_la_SOURCES=$(numtasks_sources)

instrumentation_libnanox_throttle_idlethreads_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_throttle_idlethreads_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_throttle_idlethreads_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_throttle_idlethreads_la_SOURCES=$(idlethreads_sources)

instrumentation_libnanox_throttle_taskdepth_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_throttle_taskdepth_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_throttle_taskdepth_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_throttle_taskdepth_la_SOURCES=$(taskdepth_sources)

instrumentation_libnanox_throttle_readytasks_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_throttle_readytasks_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_throttle_readytasks_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_throttle_readytasks_la_SOURCES=$(readytasks_sources)
endif

if is_instrumentation_debug_enabled
instrumentation_debug_LTLIBRARIES += \
	instrumentation-debug/libnanox-throttle-hysteresis.la \
	instrumentation-debug/libnanox-throttle-dummy.la \
	instrumentation-debug/libnanox-throttle-numtasks.la \
	instrumentation-debug/libnanox-throttle-idlethreads.la \
	instrumentation-debug/libnanox-throttle-taskdepth.la \
	instrumentation-debug/libnanox-throttle-readytasks.la \
	$(END)

instrumentation_debug_libnanox_throttle_hysteresis_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_throttle_hysteresis_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_throttle_hysteresis_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_throttle_hysteresis_la_SOURCES=$(hysteresis_sources)

instrumentation_debug_libnanox_throttle_dummy_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_throttle_dummy_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_throttle_dummy_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_throttle_dummy_la_SOURCES=$(dummy_sources)

instrumentation_debug_libnanox_throttle_numtasks_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_throttle_numtasks_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_throttle_numtasks_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_throttle_numtasks_la_SOURCES=$(numtasks_sources)

instrumentation_debug_libnanox_throttle_idlethreads_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_throttle_idlethreads_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_throttle_idlethreads_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_throttle_idlethreads_la_SOURCES=$(idlethreads_sources)

instrumentation_debug_libnanox_throttle_taskdepth_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_throttle_taskdepth_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_throttle_taskdepth_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_throttle_taskdepth_la_SOURCES=$(taskdepth_sources)

instrumentation_debug_libnanox_throttle_readytasks_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_throttle_readytasks_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_throttle_readytasks_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_throttle_readytasks_la_SOURCES=$(readytasks_sources)
endif

if is_performance_enabled
performance_LTLIBRARIES += \
	performance/libnanox-throttle-hysteresis.la \
	performance/libnanox-throttle-dummy.la \
	performance/libnanox-throttle-numtasks.la \
	performance/libnanox-throttle-idlethreads.la \
	performance/libnanox-throttle-taskdepth.la \
	performance/libnanox-throttle-readytasks.la \
	$(END)

performance_libnanox_throttle_hysteresis_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_throttle_hysteresis_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_throttle_hysteresis_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_throttle_hysteresis_la_SOURCES=$(hysteresis_sources)

performance_libnanox_throttle_dummy_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_throttle_dummy_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_throttle_dummy_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_throttle_dummy_la_SOURCES=$(dummy_sources)

performance_libnanox_throttle_numtasks_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_throttle_numtasks_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_throttle_numtasks_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_throttle_numtasks_la_SOURCES=$(numtasks_sources)

performance_libnanox_throttle_idlethreads_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_throttle_idlethreads_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_throttle_idlethreads_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_throttle_idlethreads_la_SOURCES=$(idlethreads_sources)

performance_libnanox_throttle_taskdepth_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_throttle_taskdepth_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_throttle_taskdepth_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_throttle_taskdepth_la_SOURCES=$(taskdepth_sources)

performance_libnanox_throttle_readytasks_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_throttle_readytasks_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_throttle_readytasks_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_throttle_readytasks_la_SOURCES=$(readytasks_sources)
endif
######################################################################################################
######################################################################################################

######################################################################################################
#### Barrier ########################################################################################
######################################################################################################
old_centralized_sources=\
	barr/old_centralized_barrier.cpp \
	$(END)

centralized_sources=\
	barr/centralized_barrier.cpp \
	$(END)

posix_sources=\
	barr/posix_barrier.cpp \
	$(END)

dissemination_sources=\
	barr/dissem_barrier.cpp \
	$(END)

tree_sources=\
	barr/tree_barrier.cpp \
	$(END)

if is_debug_enabled
debug_LTLIBRARIES += \
        debug/libnanox-barrier-old-centralized.la \
        debug/libnanox-barrier-centralized.la \
	$(END)

debug_libnanox_barrier_old_centralized_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_barrier_old_centralized_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_barrier_old_centralized_la_LDFLAGS=$(AM_LDFLAGS) $(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_barrier_old_centralized_la_SOURCES=$(old_centralized_sources)

debug_libnanox_barrier_centralized_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_barrier_centralized_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_barrier_centralized_la_LDFLAGS=$(AM_LDFLAGS) $(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_barrier_centralized_la_SOURCES=$(centralized_sources)
endif

if is_instrumentation_enabled
instrumentation_LTLIBRARIES += \
        instrumentation/libnanox-barrier-old-centralized.la \
        instrumentation/libnanox-barrier-centralized.la \
	$(END)

instrumentation_libnanox_barrier_old_centralized_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_barrier_old_centralized_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_barrier_old_centralized_la_LDFLAGS=$(AM_LDFLAGS) $(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_barrier_old_centralized_la_SOURCES=$(old_centralized_sources)

instrumentation_libnanox_barrier_centralized_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_barrier_centralized_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_barrier_centralized_la_LDFLAGS=$(AM_LDFLAGS) $(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_barrier_centralized_la_SOURCES=$(centralized_sources)
endif

if is_instrumentation_debug_enabled
instrumentation_debug_LTLIBRARIES += \
        instrumentation-debug/libnanox-barrier-old-centralized.la \
        instrumentation-debug/libnanox-barrier-centralized.la \
	$(END)

instrumentation_debug_libnanox_barrier_old_centralized_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_barrier_old_centralized_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_barrier_old_centralized_la_LDFLAGS=$(AM_LDFLAGS) $(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_barrier_old_centralized_la_SOURCES=$(old_centralized_sources)

instrumentation_debug_libnanox_barrier_centralized_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_barrier_centralized_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_barrier_centralized_la_LDFLAGS=$(AM_LDFLAGS) $(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_barrier_centralized_la_SOURCES=$(centralized_sources)
endif

if is_performance_enabled
performance_LTLIBRARIES += \
        performance/libnanox-barrier-old-centralized.la \
        performance/libnanox-barrier-centralized.la \
	$(END)

performance_libnanox_barrier_old_centralized_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_barrier_old_centralized_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_barrier_old_centralized_la_LDFLAGS=$(AM_LDFLAGS) $(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_barrier_old_centralized_la_SOURCES=$(old_centralized_sources)

performance_libnanox_barrier_centralized_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_barrier_centralized_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_barrier_centralized_la_LDFLAGS=$(AM_LDFLAGS) $(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_barrier_centralized_la_SOURCES=$(centralized_sources)
endif
######################################################################################################
######################################################################################################


######################################################################################################
#### Slicers #########################################################################################
######################################################################################################
static_for_sources=\
	slicers/static_for.cpp \
	$(END)

dynamic_for_sources=\
	slicers/dynamic_for.cpp \
	$(END)

guided_for_sources=\
	slicers/guided_for.cpp \
	$(END)

repeat_n_sources=\
	slicers/repeat_n.cpp \
	$(END)

compound_wd_sources=\
	slicers/compound_wd.cpp \
	$(END)

replicate_sources=\
	slicers/replicate.cpp \
	$(END)

if is_debug_enabled
debug_LTLIBRARIES += \
	debug/libnanox-slicer-static_for.la \
	debug/libnanox-slicer-dynamic_for.la \
	debug/libnanox-slicer-guided_for.la \
	debug/libnanox-slicer-repeat_n.la \
	debug/libnanox-slicer-compound_wd.la \
	debug/libnanox-slicer-replicate.la \
	$(END)

debug_libnanox_slicer_static_for_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_slicer_static_for_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_slicer_static_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_slicer_static_for_la_SOURCES=$(static_for_sources)

debug_libnanox_slicer_dynamic_for_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_slicer_dynamic_for_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_slicer_dynamic_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_slicer_dynamic_for_la_SOURCES=$(dynamic_for_sources)

debug_libnanox_slicer_guided_for_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_slicer_guided_for_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_slicer_guided_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_slicer_guided_for_la_SOURCES=$(guided_for_sources)

debug_libnanox_slicer_repeat_n_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_slicer_repeat_n_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_slicer_repeat_n_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_slicer_repeat_n_la_SOURCES=$(repeat_n_sources)

debug_libnanox_slicer_compound_wd_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_slicer_compound_wd_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_slicer_compound_wd_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_slicer_compound_wd_la_SOURCES=$(compound_wd_sources)

debug_libnanox_slicer_replicate_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_slicer_replicate_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_slicer_replicate_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_slicer_replicate_la_SOURCES=$(replicate_sources)
endif

if is_instrumentation_enabled
instrumentation_LTLIBRARIES += \
	instrumentation/libnanox-slicer-static_for.la \
	instrumentation/libnanox-slicer-dynamic_for.la \
	instrumentation/libnanox-slicer-guided_for.la \
	instrumentation/libnanox-slicer-repeat_n.la \
	instrumentation/libnanox-slicer-compound_wd.la \
	instrumentation/libnanox-slicer-replicate.la \
	$(END)

instrumentation_libnanox_slicer_static_for_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_slicer_static_for_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_slicer_static_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_slicer_static_for_la_SOURCES=$(static_for_sources)

instrumentation_libnanox_slicer_dynamic_for_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_slicer_dynamic_for_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_slicer_dynamic_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_slicer_dynamic_for_la_SOURCES=$(dynamic_for_sources)

instrumentation_libnanox_slicer_guided_for_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_slicer_guided_for_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_slicer_guided_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_slicer_guided_for_la_SOURCES=$(guided_for_sources)

instrumentation_libnanox_slicer_repeat_n_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_slicer_repeat_n_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_slicer_repeat_n_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_slicer_repeat_n_la_SOURCES=$(repeat_n_sources)

instrumentation_libnanox_slicer_compound_wd_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_slicer_compound_wd_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_slicer_compound_wd_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_slicer_compound_wd_la_SOURCES=$(compound_wd_sources)

instrumentation_libnanox_slicer_replicate_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_slicer_replicate_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_slicer_replicate_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_slicer_replicate_la_SOURCES=$(replicate_sources)
endif

if is_instrumentation_debug_enabled
instrumentation_debug_LTLIBRARIES += \
	instrumentation-debug/libnanox-slicer-static_for.la \
	instrumentation-debug/libnanox-slicer-dynamic_for.la \
	instrumentation-debug/libnanox-slicer-guided_for.la \
	instrumentation-debug/libnanox-slicer-repeat_n.la \
	instrumentation-debug/libnanox-slicer-compound_wd.la \
	instrumentation-debug/libnanox-slicer-replicate.la \
	$(END)

instrumentation_debug_libnanox_slicer_static_for_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_slicer_static_for_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_slicer_static_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_slicer_static_for_la_SOURCES=$(static_for_sources)

instrumentation_debug_libnanox_slicer_dynamic_for_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_slicer_dynamic_for_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_slicer_dynamic_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_slicer_dynamic_for_la_SOURCES=$(dynamic_for_sources)

instrumentation_debug_libnanox_slicer_guided_for_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_slicer_guided_for_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_slicer_guided_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_slicer_guided_for_la_SOURCES=$(guided_for_sources)

instrumentation_debug_libnanox_slicer_repeat_n_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_slicer_repeat_n_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_slicer_repeat_n_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_slicer_repeat_n_la_SOURCES=$(repeat_n_sources)

instrumentation_debug_libnanox_slicer_compound_wd_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_slicer_compound_wd_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_slicer_compound_wd_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_slicer_compound_wd_la_SOURCES=$(compound_wd_sources)

instrumentation_debug_libnanox_slicer_replicate_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_slicer_replicate_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_slicer_replicate_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_slicer_replicate_la_SOURCES=$(replicate_sources)
endif

if is_performance_enabled
performance_LTLIBRARIES += \
	performance/libnanox-slicer-static_for.la \
	performance/libnanox-slicer-dynamic_for.la \
	performance/libnanox-slicer-guided_for.la \
	performance/libnanox-slicer-repeat_n.la \
	performance/libnanox-slicer-compound_wd.la \
	performance/libnanox-slicer-replicate.la \
	$(END)

performance_libnanox_slicer_static_for_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_slicer_static_for_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_slicer_static_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_slicer_static_for_la_SOURCES=$(static_for_sources)

performance_libnanox_slicer_dynamic_for_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_slicer_dynamic_for_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_slicer_dynamic_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_slicer_dynamic_for_la_SOURCES=$(dynamic_for_sources)

performance_libnanox_slicer_guided_for_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_slicer_guided_for_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_slicer_guided_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_slicer_guided_for_la_SOURCES=$(guided_for_sources)

performance_libnanox_slicer_repeat_n_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_slicer_repeat_n_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_slicer_repeat_n_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_slicer_repeat_n_la_SOURCES=$(repeat_n_sources)

performance_libnanox_slicer_compound_wd_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_slicer_compound_wd_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_slicer_compound_wd_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_slicer_compound_wd_la_SOURCES=$(compound_wd_sources)

performance_libnanox_slicer_replicate_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_slicer_replicate_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_slicer_replicate_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_slicer_replicate_la_SOURCES=$(replicate_sources)
endif
######################################################################################################
######################################################################################################


######################################################################################################
#### Instrumentation #################################################################################
######################################################################################################
if instrumentation_EXTRAE
extrae_cppflags=-I@EXTRAE_INC@
extrae_ldflags=-L@EXTRAE_LIB@ -lnanostrace
endif

if instrumentation_AYUDAME
ayudame_cppflags=-I@AYUDAME_INC@ -DPREFIX='"$(prefix)"'
ayudame_ldflags=-L@AYUDAME_LIB@ -layudame
endif

if instrumentation_NEXTSIM
nextsim_cppflags=-I@NEXTSIM_INC@ -DPREFIX='"$(prefix)"'
endif

empty_sources=\
	instrumentation/empty_trace.cpp \
	$(END)

print_sources=\
	instrumentation/print_trace.cpp \
	$(END)

extrae_sources=\
	instrumentation/extrae.cpp \
	instrumentation/ompi_services.cpp \
	$(END)

ayudame_sources=\
	instrumentation/ayudame.cpp \
	$(END)

nextsim_sources=\
	instrumentation/nextsim.cpp \
	$(END)

tdg_sources=\
    instrumentation/tdg_utils.hpp \
    instrumentation/tdg.cpp \
    $(END)

ompt_sources=\
    instrumentation/ompt.cpp \
	instrumentation/ompt-headers/ompt.h \
	instrumentation/ompt-headers/ompt-api.h \
	instrumentation/ompt-headers/ompt-events.h \
	instrumentation/ompt-headers/ompt-initialize.h \
	instrumentation/ompt-headers/ompt-records.h \
	instrumentation/ompt-headers/ompt-states.h \
	instrumentation/ompt-headers/ompt-types.h \
	instrumentation/ompt-headers/ompt-types-callbacks.h \
    $(END)

if is_debug_enabled
debug_LTLIBRARIES += \
	debug/libnanox-instrumentation-empty_trace.la \
	debug/libnanox-instrumentation-print_trace.la \
	debug/libnanox-instrumentation-tdg.la \
	$(END)

if instrumentation_EXTRAE
debug_LTLIBRARIES+=debug/libnanox-instrumentation-extrae.la
endif

if instrumentation_AYUDAME
debug_LTLIBRARIES+=debug/libnanox-instrumentation-ayudame.la
endif

debug_libnanox_instrumentation_empty_trace_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_instrumentation_empty_trace_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_instrumentation_empty_trace_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_instrumentation_empty_trace_la_SOURCES=$(empty_sources)

debug_libnanox_instrumentation_print_trace_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_instrumentation_print_trace_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_instrumentation_print_trace_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_instrumentation_print_trace_la_SOURCES=$(print_sources)

debug_libnanox_instrumentation_tdg_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_instrumentation_tdg_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_instrumentation_tdg_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_instrumentation_tdg_la_SOURCES=$(tdg_sources)

if instrumentation_EXTRAE
debug_libnanox_instrumentation_extrae_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_instrumentation_extrae_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_instrumentation_extrae_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_instrumentation_extrae_la_SOURCES=$(extrae_sources)
endif

if instrumentation_AYUDAME
debug_libnanox_instrumentation_ayudame_la_CPPFLAGS=$(common_debug_CPPFLAGS) $(ayudame_cppflags)
debug_libnanox_instrumentation_ayudame_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_instrumentation_ayudame_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) $(ayudame_ldflags)
debug_libnanox_instrumentation_ayudame_la_SOURCES=$(ayudame_sources)
endif
endif

if is_instrumentation_enabled
instrumentation_LTLIBRARIES += \
	instrumentation/libnanox-instrumentation-empty_trace.la \
	instrumentation/libnanox-instrumentation-print_trace.la \
	instrumentation/libnanox-instrumentation-tdg.la \
	instrumentation/libnanox-instrumentation-ompt.la \
	$(END)

if instrumentation_EXTRAE
instrumentation_LTLIBRARIES+=instrumentation/libnanox-instrumentation-extrae.la
endif

if instrumentation_AYUDAME
instrumentation_LTLIBRARIES+=instrumentation/libnanox-instrumentation-ayudame.la
endif

if instrumentation_NEXTSIM
instrumentation_LTLIBRARIES+=instrumentation/libnanox-instrumentation-nextsim.la
endif

instrumentation_libnanox_instrumentation_empty_trace_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_instrumentation_empty_trace_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_instrumentation_empty_trace_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_instrumentation_empty_trace_la_SOURCES=$(empty_sources)

instrumentation_libnanox_instrumentation_print_trace_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_instrumentation_print_trace_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_instrumentation_print_trace_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_instrumentation_print_trace_la_SOURCES=$(print_sources)

instrumentation_libnanox_instrumentation_tdg_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_instrumentation_tdg_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_instrumentation_tdg_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_instrumentation_tdg_la_SOURCES=$(tdg_sources)

if instrumentation_EXTRAE
instrumentation_libnanox_instrumentation_extrae_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) $(extrae_cppflags)
instrumentation_libnanox_instrumentation_extrae_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_instrumentation_extrae_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) $(extrae_ldflags)
instrumentation_libnanox_instrumentation_extrae_la_SOURCES=$(extrae_sources)
endif

if instrumentation_AYUDAME
instrumentation_libnanox_instrumentation_ayudame_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) $(ayudame_cppflags)
instrumentation_libnanox_instrumentation_ayudame_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_instrumentation_ayudame_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) $(ayudame_ldflags)
instrumentation_libnanox_instrumentation_ayudame_la_LIBADD=$(top_builddir)/src/apis/instrumentation/libnanox-c.la
instrumentation_libnanox_instrumentation_ayudame_la_SOURCES=$(ayudame_sources)
instrumentation_libnanox_instrumentation_ayudame_la_DEPENDENCIES=$(top_builddir)/src/apis/instrumentation/libnanox-c.la
endif

if instrumentation_NEXTSIM
instrumentation_libnanox_instrumentation_nextsim_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) $(nextsim_cppflags)
instrumentation_libnanox_instrumentation_nextsim_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_instrumentation_nextsim_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_instrumentation_nextsim_la_SOURCES=$(nextsim_sources)
endif

instrumentation_libnanox_instrumentation_ompt_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) -I$(top_srcdir)/src/plugins/instrumentation/ompt-headers
instrumentation_libnanox_instrumentation_ompt_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_instrumentation_ompt_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_instrumentation_ompt_la_SOURCES=$(ompt_sources)

endif

if is_instrumentation_debug_enabled
instrumentation_debug_LTLIBRARIES += \
	instrumentation-debug/libnanox-instrumentation-empty_trace.la \
	instrumentation-debug/libnanox-instrumentation-print_trace.la \
	instrumentation-debug/libnanox-instrumentation-tdg.la \
	instrumentation-debug/libnanox-instrumentation-ompt.la \
	$(END)

if instrumentation_EXTRAE
instrumentation_debug_LTLIBRARIES+=instrumentation-debug/libnanox-instrumentation-extrae.la
endif

if instrumentation_AYUDAME
instrumentation_debug_LTLIBRARIES+=instrumentation-debug/libnanox-instrumentation-ayudame.la
endif

instrumentation_debug_libnanox_instrumentation_empty_trace_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_instrumentation_empty_trace_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_instrumentation_empty_trace_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_instrumentation_empty_trace_la_SOURCES=$(empty_sources)

instrumentation_debug_libnanox_instrumentation_print_trace_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_instrumentation_print_trace_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_instrumentation_print_trace_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_instrumentation_print_trace_la_SOURCES=$(print_sources)

instrumentation_debug_libnanox_instrumentation_tdg_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_instrumentation_tdg_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_instrumentation_tdg_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_instrumentation_tdg_la_SOURCES=$(tdg_sources)

if instrumentation_EXTRAE
instrumentation_debug_libnanox_instrumentation_extrae_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) $(extrae_cppflags)
instrumentation_debug_libnanox_instrumentation_extrae_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_instrumentation_extrae_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) $(extrae_ldflags)
instrumentation_debug_libnanox_instrumentation_extrae_la_SOURCES=$(extrae_sources)
endif

if instrumentation_AYUDAME
instrumentation_debug_libnanox_instrumentation_ayudame_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) $(ayudame_cppflags)
instrumentation_debug_libnanox_instrumentation_ayudame_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_instrumentation_ayudame_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) $(ayudame_ldflags)
instrumentation_debug_libnanox_instrumentation_ayudame_la_SOURCES=$(ayudame_sources)
endif

instrumentation_debug_libnanox_instrumentation_ompt_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) -I$(top_srcdir)/src/plugins/instrumentation/ompt-headers
instrumentation_debug_libnanox_instrumentation_ompt_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_instrumentation_ompt_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_instrumentation_ompt_la_SOURCES=$(ompt_sources)

endif

if is_performance_enabled
performance_LTLIBRARIES += \
	performance/libnanox-instrumentation-empty_trace.la \
	performance/libnanox-instrumentation-print_trace.la \
	performance/libnanox-instrumentation-tdg.la \
	$(END)

if instrumentation_EXTRAE
performance_LTLIBRARIES+=performance/libnanox-instrumentation-extrae.la
endif

if instrumentation_AYUDAME
performance_LTLIBRARIES+=performance/libnanox-instrumentation-ayudame.la
endif

performance_libnanox_instrumentation_empty_trace_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_instrumentation_empty_trace_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_instrumentation_empty_trace_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_instrumentation_empty_trace_la_SOURCES=$(empty_sources)

performance_libnanox_instrumentation_print_trace_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_instrumentation_print_trace_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_instrumentation_print_trace_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_instrumentation_print_trace_la_SOURCES=$(print_sources)

performance_libnanox_instrumentation_tdg_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_instrumentation_tdg_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_instrumentation_tdg_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_instrumentation_tdg_la_SOURCES=$(tdg_sources)

if instrumentation_EXTRAE
performance_libnanox_instrumentation_extrae_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_instrumentation_extrae_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_instrumentation_extrae_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_instrumentation_extrae_la_SOURCES=$(extrae_sources)
endif

if instrumentation_AYUDAME
performance_libnanox_instrumentation_ayudame_la_CPPFLAGS=$(common_performance_CPPFLAGS) $(ayudame_cppflags)
performance_libnanox_instrumentation_ayudame_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_instrumentation_ayudame_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) $(ayudame_ldflags)
performance_libnanox_instrumentation_ayudame_la_SOURCES=$(ayudame_sources)
endif
endif
######################################################################################################
######################################################################################################


######################################################################################################
#### Worksharing #####################################################################################
######################################################################################################
worksharing_static_for_sources=\
	worksharing/static.cpp \
	worksharing/loop.hpp \
	$(END)
worksharing_dynamic_for_sources=\
	worksharing/dynamic.cpp \
	worksharing/loop.hpp \
	$(END)
worksharing_guided_for_sources=\
	worksharing/guided.cpp \
	worksharing/loop.hpp \
	$(END)

if is_debug_enabled
debug_LTLIBRARIES += \
	debug/libnanox-worksharing-static_for.la \
	debug/libnanox-worksharing-dynamic_for.la \
	debug/libnanox-worksharing-guided_for.la \
	$(END)

debug_libnanox_worksharing_static_for_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_worksharing_static_for_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_worksharing_static_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_worksharing_static_for_la_SOURCES=$(worksharing_static_for_sources)

debug_libnanox_worksharing_dynamic_for_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_worksharing_dynamic_for_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_worksharing_dynamic_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_worksharing_dynamic_for_la_SOURCES=$(worksharing_dynamic_for_sources)

debug_libnanox_worksharing_guided_for_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_worksharing_guided_for_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_worksharing_guided_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_worksharing_guided_for_la_SOURCES=$(worksharing_guided_for_sources)

endif

if is_performance_enabled
performance_LTLIBRARIES += \
	performance/libnanox-worksharing-static_for.la \
	performance/libnanox-worksharing-dynamic_for.la \
	performance/libnanox-worksharing-guided_for.la \
	$(END)

performance_libnanox_worksharing_static_for_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_worksharing_static_for_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_worksharing_static_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_worksharing_static_for_la_SOURCES=$(worksharing_static_for_sources)

performance_libnanox_worksharing_dynamic_for_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_worksharing_dynamic_for_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_worksharing_dynamic_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_worksharing_dynamic_for_la_SOURCES=$(worksharing_dynamic_for_sources)

performance_libnanox_worksharing_guided_for_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_worksharing_guided_for_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_worksharing_guided_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_worksharing_guided_for_la_SOURCES=$(worksharing_guided_for_sources)

endif

if is_instrumentation_enabled
instrumentation_LTLIBRARIES += \
	instrumentation/libnanox-worksharing-static_for.la \
	instrumentation/libnanox-worksharing-dynamic_for.la \
	instrumentation/libnanox-worksharing-guided_for.la \
	$(END)

instrumentation_libnanox_worksharing_static_for_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_worksharing_static_for_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_worksharing_static_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_worksharing_static_for_la_SOURCES=$(worksharing_static_for_sources)

instrumentation_libnanox_worksharing_dynamic_for_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_worksharing_dynamic_for_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_worksharing_dynamic_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_worksharing_dynamic_for_la_SOURCES=$(worksharing_dynamic_for_sources)

instrumentation_libnanox_worksharing_guided_for_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_worksharing_guided_for_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_worksharing_guided_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_worksharing_guided_for_la_SOURCES=$(worksharing_guided_for_sources)

endif

if is_instrumentation_debug_enabled
instrumentation_debug_LTLIBRARIES += \
	instrumentation-debug/libnanox-worksharing-static_for.la \
	instrumentation-debug/libnanox-worksharing-dynamic_for.la \
	instrumentation-debug/libnanox-worksharing-guided_for.la \
	$(END)

instrumentation_debug_libnanox_worksharing_static_for_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_worksharing_static_for_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_worksharing_static_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_worksharing_static_for_la_SOURCES=$(worksharing_static_for_sources)

instrumentation_debug_libnanox_worksharing_dynamic_for_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_worksharing_dynamic_for_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_worksharing_dynamic_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_worksharing_dynamic_for_la_SOURCES=$(worksharing_dynamic_for_sources)

instrumentation_debug_libnanox_worksharing_guided_for_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_worksharing_guided_for_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_worksharing_guided_for_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_worksharing_guided_for_la_SOURCES=$(worksharing_guided_for_sources)

endif
######################################################################################################
######################################################################################################


######################################################################################################
#### Deps ############################################################################################
######################################################################################################
plain_sources=\
	deps/plain_deps.cpp \
	deps/basedependenciesdomain_decl.hpp \
	deps/basedependenciesdomain.hpp \
	$(END)

regions_sources=\
   deps/regions_deps.cpp \
   deps/basedependenciesdomain_decl.hpp \
   deps/basedependenciesdomain.hpp \
   deps/baseregionsdependenciesdomain_decl.hpp \
   deps/baseregionsdependenciesdomain.hpp \
   $(END)

perfect_regions_sources=\
	deps/perfect_regions_deps.cpp \
	deps/basedependenciesdomain_decl.hpp \
	deps/basedependenciesdomain.hpp \
	$(END)
		
cregions_sources=\
	deps/cregions_deps.cpp \
	deps/basedependenciesdomain_decl.hpp \
	deps/basedependenciesdomain.hpp \
	$(END)
	
	
cregions_nocache_sources=\
	deps/cregions_deps_nocache.cpp \
	deps/basedependenciesdomain_decl.hpp \
	deps/basedependenciesdomain.hpp \
	$(END)

if is_debug_enabled
debug_LTLIBRARIES += \
        debug/libnanox-deps-plain.la\
        debug/libnanox-deps-perfect-regions.la\
        debug/libnanox-deps-regions.la\
        debug/libnanox-deps-cregions.la\
        debug/libnanox-deps-cregions_nocache.la\
	$(END)

debug_libnanox_deps_plain_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_deps_plain_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_deps_plain_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_deps_plain_la_SOURCES=$(plain_sources)

debug_libnanox_deps_perfect_regions_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_deps_perfect_regions_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_deps_perfect_regions_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_deps_perfect_regions_la_SOURCES=$(perfect_regions_sources)

debug_libnanox_deps_regions_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_deps_regions_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_deps_regions_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_deps_regions_la_SOURCES=$(regions_sources)

debug_libnanox_deps_cregions_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_deps_cregions_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_deps_cregions_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_deps_cregions_la_SOURCES=$(cregions_sources)


debug_libnanox_deps_cregions_nocache_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libnanox_deps_cregions_nocache_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_deps_cregions_nocache_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
debug_libnanox_deps_cregions_nocache_la_SOURCES=$(cregions_nocache_sources)

endif

if is_performance_enabled
performance_LTLIBRARIES += \
	performance/libnanox-deps-plain.la \
   performance/libnanox-deps-perfect-regions.la\
   performance/libnanox-deps-regions.la\
   performance/libnanox-deps-cregions.la\
   performance/libnanox-deps-cregions_nocache.la\
	$(END)

performance_libnanox_deps_plain_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_deps_plain_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_deps_plain_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_deps_plain_la_SOURCES=$(plain_sources)

performance_libnanox_deps_perfect_regions_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_deps_perfect_regions_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_deps_perfect_regions_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_deps_perfect_regions_la_SOURCES=$(perfect_regions_sources)

performance_libnanox_deps_regions_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_deps_regions_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_deps_regions_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_deps_regions_la_SOURCES=$(regions_sources)

performance_libnanox_deps_cregions_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_deps_cregions_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_deps_cregions_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_deps_cregions_la_SOURCES=$(cregions_sources)

performance_libnanox_deps_cregions_nocache_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libnanox_deps_cregions_nocache_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_deps_cregions_nocache_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
performance_libnanox_deps_cregions_nocache_la_SOURCES=$(cregions_nocache_sources)

endif

if is_instrumentation_enabled
instrumentation_LTLIBRARIES += \
   instrumentation/libnanox-deps-plain.la\
   instrumentation/libnanox-deps-perfect-regions.la\
   instrumentation/libnanox-deps-regions.la\
   instrumentation/libnanox-deps-cregions.la\
   instrumentation/libnanox-deps-cregions_nocache.la\
	$(END)

instrumentation_libnanox_deps_plain_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_deps_plain_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_deps_plain_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_deps_plain_la_SOURCES=$(plain_sources)

instrumentation_libnanox_deps_perfect_regions_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_deps_perfect_regions_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_deps_perfect_regions_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_deps_perfect_regions_la_SOURCES=$(perfect_regions_sources)

instrumentation_libnanox_deps_regions_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_deps_regions_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_deps_regions_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_deps_regions_la_SOURCES=$(regions_sources)

instrumentation_libnanox_deps_cregions_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_deps_cregions_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_deps_cregions_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_deps_cregions_la_SOURCES=$(cregions_sources)

instrumentation_libnanox_deps_cregions_nocache_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libnanox_deps_cregions_nocache_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_deps_cregions_nocache_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_libnanox_deps_cregions_nocache_la_SOURCES=$(cregions_nocache_sources)
endif

if is_instrumentation_debug_enabled
instrumentation_debug_LTLIBRARIES += \
   instrumentation-debug/libnanox-deps-plain.la\
   instrumentation-debug/libnanox-deps-perfect-regions.la\
   instrumentation-debug/libnanox-deps-regions.la\
   instrumentation-debug/libnanox-deps-cregions.la\
   instrumentation-debug/libnanox-deps-cregions_nocache.la\
	$(END)

instrumentation_debug_libnanox_deps_plain_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_deps_plain_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_deps_plain_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_deps_plain_la_SOURCES=$(plain_sources)

instrumentation_debug_libnanox_deps_perfect_regions_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_deps_perfect_regions_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_deps_perfect_regions_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_deps_perfect_regions_la_SOURCES=$(perfect_regions_sources)

instrumentation_debug_libnanox_deps_regions_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_deps_regions_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_deps_regions_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_deps_regions_la_SOURCES=$(regions_sources)

instrumentation_debug_libnanox_deps_cregions_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_deps_cregions_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_deps_cregions_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_deps_cregions_la_SOURCES=$(cregions_sources)

instrumentation_debug_libnanox_deps_cregions_nocache_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libnanox_deps_cregions_nocache_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_deps_cregions_nocache_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags)
instrumentation_debug_libnanox_deps_cregions_nocache_la_SOURCES=$(cregions_nocache_sources)

endif
######################################################################################################
######################################################################################################
