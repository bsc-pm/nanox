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

devinclude_HEADERS = \
	os.hpp \
	pthread_decl.hpp \
	pthread.hpp \
	cpuset.hpp \
	mutex.hpp \
	condition_variable.hpp \
	filelock.hpp \
	$(END) 

os_sources = \
	cpuset.cpp \
	cpuset.hpp \
	os.hpp \
	os.cpp \
	osallocator_decl.hpp \
	osallocator.cpp \
	pthread_decl.hpp \
	pthread.hpp \
	pthread.cpp \
	$(END) 

common_libadd = \
	-lrt \
	-ldl \
	-lpthread \
	$(END)

noinst_LTLIBRARIES =


if is_debug_enabled
noinst_LTLIBRARIES += debug/libos.la


debug_libos_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libos_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libos_la_SOURCES=$(os_sources)
debug_libos_la_LIBADD=$(common_libadd)
endif

if is_instrumentation_enabled
noinst_LTLIBRARIES += instrumentation/libos.la


instrumentation_libos_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libos_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libos_la_SOURCES=$(os_sources)
instrumentation_libos_la_LIBADD=$(common_libadd)
endif

if is_instrumentation_debug_enabled
noinst_LTLIBRARIES += instrumentation-debug/libos.la


instrumentation_debug_libos_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libos_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libos_la_SOURCES=$(os_sources)
instrumentation_debug_libos_la_LIBADD=$(common_libadd)
endif

if is_performance_enabled
noinst_LTLIBRARIES += performance/libos.la


performance_libos_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libos_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libos_la_SOURCES=$(os_sources)
performance_libos_la_LIBADD=$(common_libadd)
endif
