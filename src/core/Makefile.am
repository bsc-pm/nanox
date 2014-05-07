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

#SUBDIRS = performance debug instrumentation instrumentation-debug

include ../common.am

pkginclude_HEADERS=nanos-int.h nanos_error.h

common_libadd=\
	$(END)

devinclude_HEADERS = \
	nanos_error.h \
	barrier_fwd.hpp \
	threadteam_fwd.hpp \
	basethread_fwd.hpp\
	processingelement_fwd.hpp \
	accelerator_fwd.hpp \
	schedule_fwd.hpp \
	system_fwd.hpp \
	wddeque_fwd.hpp \
	workdescriptor_fwd.hpp \
	slicer_fwd.hpp \
	instrumentationcontext_fwd.hpp \
	instrumentation_fwd.hpp \
	throttle_fwd.hpp \
	dataaccess_fwd.hpp \
	copydata_fwd.hpp \
	dependableobject_fwd.hpp \
	dependableobjectwd_fwd.hpp \
	trackableobject_fwd.hpp \
	dependenciesdomain_fwd.hpp \
	synchronizedcondition_fwd.hpp \
	address_fwd.hpp \
	basedependency_fwd.hpp \
	nanos-int.h \
	barrier_decl.hpp \
	barrier.hpp \
   cachedaccelerator_decl.hpp \
   cachedaccelerator.hpp \
	threadteam_decl.hpp \
	threadteam.hpp \
	basethread_decl.hpp\
	basethread.hpp\
	processingelement_decl.hpp \
	processingelement.hpp \
	accelerator_decl.hpp \
	schedule_decl.hpp \
	schedule.hpp \
	system_decl.hpp\
	system.hpp \
	wddeque_decl.hpp \
	wddeque.hpp \
	workdescriptor_decl.hpp \
	workdescriptor.hpp \
	worksharing_decl.hpp \
	slicer_decl.hpp \
	slicer.hpp \
	instrumentationcontext_decl.hpp \
	instrumentationcontext.hpp \
	instrumentationmodule_decl.hpp \
	instrumentation_decl.hpp \
	instrumentation.hpp \
	throttle_decl.hpp \
	dataaccess_decl.hpp \
	dataaccess.hpp \
	copydata_decl.hpp \
	copydata.hpp \
	dependableobject_decl.hpp \
	dependableobject.hpp \
	dependableobjectwd_decl.hpp \
	dependableobjectwd.hpp \
	commutationdepobj_decl.hpp \
	commutationdepobj.hpp \
	trackableobject_decl.hpp \
	trackableobject.hpp \
	dependenciesdomain_decl.hpp \
	dependenciesdomain.hpp \
	synchronizedcondition_decl.hpp \
	synchronizedcondition.hpp \
	pminterface_decl.hpp  \
	basethread_decl.hpp \
	address_decl.hpp \
	address.hpp \
	basedependency_decl.hpp \
	taskexecutionexception_decl.hpp \
	taskexecutionexception.hpp \
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
	newregiondirectory.cpp  \
	newregiondirectory.hpp  \
	newregiondirectory_decl.hpp  \
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
	memcachecopy.cpp \
	memoryops_decl.hpp \
	memoryops_fwd.hpp \
	memoryops.cpp \
	globalregt_decl.hpp \
	globalregt.cpp \
	address_fwd.hpp \
	address_decl.hpp \
	address.hpp \
	basedependency_fwd.hpp \
	basedependency_decl.hpp \
	graphentry_decl.hpp \
	taskexecutionexception_decl.hpp \
	taskexecutionexception.hpp \
	taskexecutionexception.cpp \
	$(END)

instr_sources = \
	instrumentation.cpp \
	instrumentationcontext.cpp \
	$(END)


debug_LTLIBRARIES=
instrumentation_LTLIBRARIES=
instrumentation_debug_LTLIBRARIES=
performance_LTLIBRARIES=

if DEBUG_VERSION
debug_LTLIBRARIES+=debug/libnanox.la

debug_libnanox_la_CFLAGS=$(common_cflags) $(common_debug_cflags)
debug_libnanox_la_CXXFLAGS=$(common_includes) $(common_debug_cxxflags) 
debug_libnanox_la_LDFLAGS=$(version_flags) @NANOS_CONFIG_LIBDIRS@ -Wl,-rpath,$(debugdir) $(ignore_muldefs) $(hwloc_ldflags)
debug_libnanox_la_LIBADD=$(common_libadd) @NANOS_CONFIG_LIBS_DEBUG@
debug_libnanox_la_SOURCES=$(common_sources)
debug_libnanox_la_DEPENDENCIES=$(common_libadd) @NANOS_CONFIG_LIBS_DEBUG@
endif


if INSTRUMENTATION_VERSION
instrumentation_LTLIBRARIES+=instrumentation/libnanox.la

instrumentation_libnanox_la_CFLAGS=$(common_cflags) $(common_instr_cflags)
instrumentation_libnanox_la_CXXFLAGS=$(common_includes) $(common_instr_cxxflags)
instrumentation_libnanox_la_LDFLAGS=$(version_flags) @NANOS_CONFIG_LIBDIRS@ -Wl,-rpath,$(instrumentationdir) $(ignore_muldefs) $(hwloc_ldflags)
instrumentation_libnanox_la_LIBADD=$(common_libadd) @NANOS_CONFIG_LIBS_INSTR@
instrumentation_libnanox_la_SOURCES=$(common_sources) $(instr_sources)
instrumentation_libnanox_la_DEPENDENCIES=$(common_libadd) @NANOS_CONFIG_LIBS_INSTR@
endif

if INSTRUMENTATION_DEBUG_VERSION
instrumentation_debug_LTLIBRARIES+=instrumentation-debug/libnanox.la

instrumentation_debug_libnanox_la_CFLAGS=$(common_cflags) $(common_instrdebug_cflags)
instrumentation_debug_libnanox_la_CXXFLAGS=$(common_includes) $(common_instrdebug_cxxflags)
instrumentation_debug_libnanox_la_LDFLAGS=$(version_flags) @NANOS_CONFIG_LIBDIRS@ -Wl,-rpath,$(instrumentation_debugdir) $(ignore_muldefs) $(hwloc_ldflags)
instrumentation_debug_libnanox_la_LIBADD=$(common_libadd) @NANOS_CONFIG_LIBS_INSTRDEBUG@
instrumentation_debug_libnanox_la_SOURCES=$(common_sources) $(instr_sources)
instrumentation_debug_libnanox_la_DEPENDENCIES=$(common_libadd) @NANOS_CONFIG_LIBS_INSTRDEBUG@
endif

if PERFORMANCE_VERSION
performance_LTLIBRARIES+=performance/libnanox.la

performance_libnanox_la_CFLAGS=$(common_cflags) $(common_perfo_cflags)
performance_libnanox_la_CXXFLAGS=$(common_includes) $(common_perfo_cxxflags) 
performance_libnanox_la_LDFLAGS=$(version_flags) @NANOS_CONFIG_LIBDIRS@ -Wl,-rpath,$(performancedir) $(ignore_muldefs) $(hwloc_ldflags)
performance_libnanox_la_LIBADD=$(common_libadd) @NANOS_CONFIG_LIBS_PERFO@
performance_libnanox_la_SOURCES=$(common_sources)
performance_libnanox_la_DEPENDENCIES=$(common_libadd) @NANOS_CONFIG_LIBS_PERFO@
endif

