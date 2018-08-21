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

#SUBDIRS = performance debug instrumentation instrumentation-debug

include $(top_srcdir)/src/common.am

pkginclude_HEADERS=nanos-int.h nanos_error.h

devinclude_HEADERS = \
	nanos-int.h \
	nanos_error.h \
	barrier_fwd.hpp \
	barrier_decl.hpp \
	barrier.hpp \
	cachedaccelerator_decl.hpp \
	cachedaccelerator.hpp \
	threadmanager_decl.hpp \
	threadteam_fwd.hpp \
	threadteam_decl.hpp \
	threadteam.hpp \
	basethread_fwd.hpp\
	basethread_decl.hpp\
	basethread.hpp\
	processingelement_fwd.hpp \
	processingelement_decl.hpp \
	processingelement.hpp \
	accelerator_fwd.hpp \
	accelerator_decl.hpp \
	schedule_fwd.hpp \
	schedule_decl.hpp \
	schedule.hpp \
	system_fwd.hpp \
	system_decl.hpp\
	system.hpp \
	wddeque_fwd.hpp \
	wddeque_decl.hpp \
	wddeque.hpp \
	workdescriptor_fwd.hpp \
	workdescriptor_decl.hpp \
	workdescriptor.hpp \
	worksharing_decl.hpp \
	slicer_fwd.hpp \
	slicer_decl.hpp \
	slicer.hpp \
	instrumentationcontext_fwd.hpp \
	instrumentationcontext_decl.hpp \
	instrumentationcontext.hpp \
	instrumentationmodule_decl.hpp \
	instrumentation_fwd.hpp \
	instrumentation_decl.hpp \
	instrumentation.hpp \
	throttle_fwd.hpp \
	throttle_decl.hpp \
	dataaccess_fwd.hpp \
	dataaccess_decl.hpp \
	dataaccess.hpp \
	copydata_fwd.hpp \
	copydata_decl.hpp \
	copydata.hpp \
	dependableobject_fwd.hpp \
	dependableobject_decl.hpp \
	dependableobject.hpp \
	dependableobjectwd_fwd.hpp \
	dependableobjectwd_decl.hpp \
	dependableobjectwd.hpp \
	commutationdepobj_fwd.hpp \
	commutationdepobj_decl.hpp \
	commutationdepobj.hpp \
	dependenciesdomain_fwd.hpp \
	dependenciesdomain_decl.hpp \
	dependenciesdomain.hpp \
	synchronizedcondition_fwd.hpp \
	synchronizedcondition_decl.hpp \
	synchronizedcondition.hpp \
	pminterface_decl.hpp  \
	basethread_decl.hpp \
	asyncthread_fwd.hpp \
	asyncthread_decl.hpp \
	asyncthread.hpp \
	networkapi.hpp  \
	network_decl.hpp  \
	bitcounter.hpp \
	regiondict_decl.hpp  \
	regiondict.hpp  \
	regiondirectory.hpp  \
	regiondirectory_decl.hpp  \
	regioncache_fwd.hpp  \
	regioncache_decl.hpp  \
	regioncache.hpp  \
	location_decl.hpp  \
	location.hpp  \
	deviceops_decl.hpp  \
	deviceops_fwd.hpp  \
	deviceops.hpp  \
	version.hpp  \
	version_decl.hpp  \
	cachedregionstatus.hpp  \
	cachedregionstatus_decl.hpp  \
	addressspace_decl.hpp  \
	addressspace.hpp  \
	addressspace_fwd.hpp  \
	memcontroller_fwd.hpp  \
	memcontroller_decl.hpp  \
	memcachecopy_decl.hpp \
	memcachecopy.hpp \
	memcachecopy_fwd.hpp \
	memoryops_decl.hpp \
	memoryops_fwd.hpp \
	globalregt_decl.hpp \
	globalregt.hpp \
	address_fwd.hpp \
	address_decl.hpp \
	address.hpp \
	basedependency_fwd.hpp \
	basedependency_decl.hpp \
	graphentry_decl.hpp \
	taskexecutionexception_decl.hpp \
	taskexecutionexception.hpp \
	trackableobject_fwd.hpp \
	trackableobject_decl.hpp \
	trackableobject.hpp \
	regionset_decl.hpp \
	router_fwd.hpp \
	router_decl.hpp \
	router.hpp \
	invalidationcontroller_decl.hpp \
	invalidationcontroller_fwd.hpp \
	task_reduction_decl.hpp \
	task_reduction.hpp \
	$(END)

common_sources=\
	nanos-int.h \
	nanos_error.h \
	barrier_fwd.hpp \
	barrier_decl.hpp \
	barrier.hpp \
	cachedaccelerator_decl.hpp \
	cachedaccelerator.hpp \
	threadteam_fwd.hpp \
	threadteam_decl.hpp \
	threadteam.hpp \
	threadteam.cpp \
	basethread_fwd.hpp\
	basethread_decl.hpp\
	basethread.hpp\
	basethread.cpp\
	asyncthread_fwd.hpp \
	asyncthread_decl.hpp \
	asyncthread.hpp \
	asyncthread.cpp \
	processingelement_fwd.hpp \
	processingelement_decl.hpp \
	processingelement.hpp \
	processingelement.cpp \
	accelerator_fwd.hpp \
	accelerator_decl.hpp \
	accelerator.cpp \
	schedule_fwd.hpp \
	schedule_decl.hpp \
	schedule.hpp \
	schedule.cpp \
	system_fwd.hpp \
	system_decl.hpp\
	system.hpp \
	system.cpp \
	wddeque_fwd.hpp \
	wddeque_decl.hpp \
	wddeque.hpp \
	workdescriptor_fwd.hpp \
	workdescriptor_decl.hpp \
	workdescriptor.hpp \
	workdescriptor.cpp \
	worksharing_decl.hpp \
	slicer_fwd.hpp \
	slicer_decl.hpp \
	slicer.hpp \
	instrumentationcontext_fwd.hpp \
	instrumentationcontext_decl.hpp \
	instrumentationcontext.hpp \
	instrumentationmodule_decl.hpp \
	instrumentation_fwd.hpp \
	instrumentation_decl.hpp \
	instrumentation.hpp \
	throttle_fwd.hpp \
	throttle_decl.hpp \
	dataaccess_fwd.hpp \
	dataaccess_decl.hpp \
	dataaccess.hpp \
	copydata_fwd.hpp \
	copydata_decl.hpp \
	copydata.hpp \
	copydata.cpp \
	dependableobject_fwd.hpp \
	dependableobject_decl.hpp \
	dependableobject.hpp \
	dependableobject.cpp \
	dependableobjectwd_fwd.hpp \
	dependableobjectwd_decl.hpp \
	dependableobjectwd.hpp \
	dependableobjectwd.cpp \
	commutationdepobj_decl.hpp \
	commutationdepobj.hpp \
	dependenciesdomain_fwd.hpp \
	dependenciesdomain_decl.hpp \
	dependenciesdomain.hpp \
	dependenciesdomain.cpp \
	synchronizedcondition_fwd.hpp \
	synchronizedcondition_decl.hpp \
	synchronizedcondition.hpp \
	pminterface_decl.hpp  \
	basethread_decl.hpp \
	networkapi.hpp  \
	network_decl.hpp  \
	network.cpp  \
	bitcounter.hpp \
	dataaccess_fwd.hpp \
	dataaccess_decl.hpp \
	dataaccess.hpp \
	regiondict_decl.hpp  \
	regiondict.hpp  \
	regiondict.cpp  \
	regiondirectory.cpp  \
	regiondirectory.hpp  \
	regiondirectory_decl.hpp  \
	regioncache_fwd.hpp  \
	regioncache_decl.hpp  \
	regioncache.hpp  \
	regioncache.cpp  \
	location_decl.hpp  \
	location.hpp  \
	deviceops_decl.hpp  \
	deviceops_fwd.hpp  \
	deviceops.hpp  \
	version.hpp  \
	version_decl.hpp  \
	cachedregionstatus.hpp  \
	cachedregionstatus_decl.hpp  \
	addressspace.cpp  \
	addressspace_decl.hpp  \
	addressspace.hpp  \
	addressspace_fwd.hpp  \
	memcontroller_fwd.hpp  \
	memcontroller_decl.hpp  \
	memcontroller.cpp  \
	memcachecopy_decl.hpp \
	memcachecopy_fwd.hpp \
	memcachecopy.hpp \
	memcachecopy.cpp \
	memoryops_decl.hpp \
	memoryops_fwd.hpp \
	memoryops.cpp \
	globalregt_decl.hpp \
	globalregt.hpp \
	globalregt.cpp \
	address_fwd.hpp \
	address_decl.hpp \
	address.hpp \
	address_fwd.hpp \
	address_decl.hpp \
	address.hpp \
	basedependency_fwd.hpp \
	basedependency_decl.hpp \
	graphentry_decl.hpp \
	taskexecutionexception_decl.hpp \
	taskexecutionexception.hpp \
	taskexecutionexception.cpp \
	regionset_decl.hpp \
	regionset.cpp \
	invalidationcontroller_decl.hpp \
	invalidationcontroller_fwd.hpp \
	invalidationcontroller.cpp \
	threadmanager_decl.hpp \
	threadmanager.cpp \
   task_reduction_decl.hpp \
   task_reduction.hpp \
	$(END)

instr_sources = \
	instrumentation.cpp \
	instrumentationcontext.cpp \
	$(END)

common_core_cppflags = @dlbcppflags@
common_core_ldflags = $(AM_LDFLAGS) $(version_flags) @dlbldflags@ -Wl,-enable-new-dtags

debug_LTLIBRARIES=
instrumentation_LTLIBRARIES=
instrumentation_debug_LTLIBRARIES=
performance_LTLIBRARIES=

if is_debug_enabled
debug_LTLIBRARIES+=debug/libnanox.la

debug_libnanox_la_CPPFLAGS=$(common_debug_CPPFLAGS) $(common_core_cppflags)
debug_libnanox_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_la_LDFLAGS=$(common_core_ldflags) -Wl,-rpath,$(libdir)/debug
debug_libnanox_la_LIBADD=$(common_libadd) @dlblibs@ @nanos_config_libs_debug@
debug_libnanox_la_SOURCES=$(common_sources)
debug_libnanox_la_DEPENDENCIES=$(common_libadd) @nanos_config_libs_debug@
endif

if is_instrumentation_debug_enabled
instrumentation_debug_LTLIBRARIES+=instrumentation-debug/libnanox.la

instrumentation_debug_libnanox_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) $(common_core_cppflags)
instrumentation_debug_libnanox_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_la_LDFLAGS=$(common_core_ldflags) -Wl,-rpath,$(libdir)/instrumentation-debug
instrumentation_debug_libnanox_la_LIBADD=$(common_libadd) @dlblibs@ @nanos_config_libs_instrumentation_debug@
instrumentation_debug_libnanox_la_SOURCES=$(common_sources) $(instr_sources)
instrumentation_debug_libnanox_la_DEPENDENCIES=$(common_libadd) @nanos_config_libs_instrumentation_debug@
endif

if is_instrumentation_enabled
instrumentation_LTLIBRARIES+=instrumentation/libnanox.la

instrumentation_libnanox_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) $(common_core_cppflags)
instrumentation_libnanox_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_la_LDFLAGS=$(common_core_ldflags) -Wl,-rpath,$(libdir)/instrumentation
instrumentation_libnanox_la_LIBADD=$(common_libadd) @dlblibs@ @nanos_config_libs_instrumentation@
instrumentation_libnanox_la_SOURCES=$(common_sources) $(instr_sources)
instrumentation_libnanox_la_DEPENDENCIES=$(common_libadd) @nanos_config_libs_instrumentation@
endif

if is_performance_enabled
performance_LTLIBRARIES+=performance/libnanox.la

performance_libnanox_la_CPPFLAGS=$(common_performance_CPPFLAGS) $(common_core_cppflags)
performance_libnanox_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_la_LDFLAGS=$(common_core_ldflags) -Wl,-rpath,$(libdir)/performance
performance_libnanox_la_LIBADD=$(common_libadd) @dlblibs@ @nanos_config_libs_performance@
performance_libnanox_la_SOURCES=$(common_sources)
performance_libnanox_la_DEPENDENCIES=$(common_libadd) @nanos_config_libs_performance@
endif

