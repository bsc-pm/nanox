#####################################################################################
#      Copyright 2015 Barcelona Supercomputing Center                               #
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

devinclude_HEADERS = \
	os.hpp \
	pthread_decl.hpp \
	pthread.hpp \
   cpuset.hpp \
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


if debug_enabled
noinst_LTLIBRARIES += debug/libos.la

nanos_debug_cxxflags=$(common_includes) $(common_debug_cxxflags) 

debug_libos_la_CXXFLAGS=$(nanos_debug_cxxflags)
debug_libos_la_SOURCES=$(os_sources)
debug_libos_la_LIBADD=$(common_libadd)
endif

if instrumentation_enabled
noinst_LTLIBRARIES += instrumentation/libos.la

nanos_instr_cxxflags=$(common_includes) $(common_instr_cxxflags) 

instrumentation_libos_la_CXXFLAGS=$(nanos_instr_cxxflags)
instrumentation_libos_la_SOURCES=$(os_sources)
instrumentation_libos_la_LIBADD=$(common_libadd)
endif

if instrumentation_debug_enabled
noinst_LTLIBRARIES += instrumentation-debug/libos.la

nanos_instrdebug_cxxflags=$(common_includes) $(common_instrdebug_cxxflags) 

instrumentation_debug_libos_la_CXXFLAGS=$(nanos_instrdebug_cxxflags)
instrumentation_debug_libos_la_SOURCES=$(os_sources)
instrumentation_debug_libos_la_LIBADD=$(common_libadd)
endif

if performance_enabled
noinst_LTLIBRARIES += performance/libos.la

nanos_perfo_cxxflags=$(common_includes) $(common_perfo_cxxflags) 

performance_libos_la_CXXFLAGS=$(nanos_perfo_cxxflags)
performance_libos_la_SOURCES=$(os_sources)
performance_libos_la_LIBADD=$(common_libadd)
endif
