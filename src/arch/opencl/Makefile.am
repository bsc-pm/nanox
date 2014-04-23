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

pkginclude_HEADERS=\
	nanos-opencl.h \
	$(END)
	
EXTRA_DIST = \
	$(END)

devinclude_HEADERS = \
    openclconfig.hpp \
    opencldd.hpp \
    opencldevice_decl.hpp \
    opencldevice.hpp \
    openclprocessor_decl.hpp \
    openclprocessor.hpp \
    openclcache.hpp \
    openclthread_decl.hpp \
    openclthread.hpp \
    openclutils.hpp \
    sharedmemallocator.hpp \
	$(END) 

opencl_api_sources = \
	nanos-opencl.h \
	opencl-api.cpp \
	$(END)
opencl_sources = \
    openclconfig.hpp \
    openclconfig.cpp \
    opencldd.hpp \
    opencldd.cpp \
    opencldevice_decl.hpp \
    opencldevice.hpp \
    openclprocessor_fwd.hpp \
    openclprocessor_decl.hpp \
    openclprocessor.hpp \
    openclprocessor.cpp \
    openclcache.hpp \
    openclcache.cpp \
    openclthread_decl.hpp \
    openclthread.hpp \
    openclthread.cpp \
    openclutils.hpp \
    sharedmemallocator.cpp \
    sharedmemallocator.hpp \
    $(END)

pe_opencl_sources = \
	openclplugin.cpp \
	$(END)

opencl_api_flags = -I$(srcdir)/../../apis/c
opencl_ldflags = -L@OPENCL_LD@ -lOpenCL
opencl_includes = -isystem @OPENCL_INC@

noinst_LTLIBRARIES =
debug_LTLIBRARIES =
instrumentation_LTLIBRARIES =
instrumentation_debug_LTLIBRARIES =
performance_LTLIBRARIES =

#TODO: Fix instrumentation and debug makefiles so they are named opencl instead of opencl and they can use headers in this folder

if DEBUG_VERSION
noinst_LTLIBRARIES += debug/libopencl.la
debug_LTLIBRARIES += debug/libnanox-opencl-api.la debug/libnanox-pe-opencl.la

nanos_debug_cxxflags=$(common_includes) $(common_debug_cxxflags) $(opencl_includes)

debug_libopencl_la_CXXFLAGS=$(nanos_debug_cxxflags) $(opencl_api_flags)
debug_libopencl_la_LDFLAGS=$(opencl_ldflags)
debug_libopencl_la_SOURCES=$(opencl_sources)

debug_libnanox_opencl_api_la_CXXFLAGS=$(nanos_debug_cxxflags) $(opencl_api_flags)
debug_libnanox_opencl_api_la_LDFLAGS=$(version_flags)
debug_libnanox_opencl_api_la_SOURCES=$(opencl_api_sources)

debug_libnanox_pe_opencl_la_CXXFLAGS=$(nanos_debug_cxxflags)
debug_libnanox_pe_opencl_la_LDFLAGS=$(ld_plugin_flags) $(opencl_ldflags)
debug_libnanox_pe_opencl_la_SOURCES=$(pe_opencl_sources)
endif

if INSTRUMENTATION_VERSION
noinst_LTLIBRARIES += instrumentation/libopencl.la
instrumentation_LTLIBRARIES += instrumentation/libnanox-opencl-api.la instrumentation/libnanox-pe-opencl.la

nanos_instr_cxxflags=$(common_includes) $(common_instr_cxxflags) $(opencl_includes)

instrumentation_libopencl_la_CXXFLAGS=$(nanos_instr_cxxflags) $(opencl_api_flags)
instrumentation_libopencl_la_LDFLAGS=$(opencl_ldflags)
instrumentation_libopencl_la_SOURCES=$(opencl_sources)

instrumentation_libnanox_opencl_api_la_CXXFLAGS=$(nanos_instr_cxxflags) $(opencl_api_flags)
instrumentation_libnanox_opencl_api_la_LDFLAGS=$(version_flags)
instrumentation_libnanox_opencl_api_la_SOURCES=$(opencl_api_sources)

instrumentation_libnanox_pe_opencl_la_CXXFLAGS=$(nanos_instr_cxxflags)
instrumentation_libnanox_pe_opencl_la_LDFLAGS=$(ld_plugin_flags) $(opencl_ldflags)
instrumentation_libnanox_pe_opencl_la_SOURCES=$(pe_opencl_sources)
endif

if INSTRUMENTATION_DEBUG_VERSION
noinst_LTLIBRARIES += instrumentation-debug/libopencl.la
instrumentation_debug_LTLIBRARIES += instrumentation-debug/libnanox-opencl-api.la instrumentation-debug/libnanox-pe-opencl.la

nanos_instrdebug_cxxflags=$(common_includes) $(common_instrdebug_cxxflags) \
                          $(opencl_includes)

instrumentation_debug_libopencl_la_CXXFLAGS=$(nanos_instrdebug_cxxflags) $(opencl_api_flags)
instrumentation_debug_libopencl_la_LDFLAGS=$(opencl_ldflags)
instrumentation_debug_libopencl_la_SOURCES=$(opencl_sources)

instrumentation_debug_libnanox_opencl_api_la_CXXFLAGS=$(nanos_instrdebug_cxxflags) $(opencl_api_flags)
instrumentation_debug_libnanox_opencl_api_la_LDFLAGS=$(version_flags)
instrumentation_debug_libnanox_opencl_api_la_SOURCES=$(opencl_api_sources)

instrumentation_debug_libnanox_pe_opencl_la_CXXFLAGS=$(nanos_instrdebug_cxxflags)
instrumentation_debug_libnanox_pe_opencl_la_LDFLAGS=$(ld_plugin_flags) $(opencl_ldflags)
instrumentation_debug_libnanox_pe_opencl_la_SOURCES=$(pe_opencl_sources)
endif

if PERFORMANCE_VERSION
noinst_LTLIBRARIES += performance/libopencl.la
performance_LTLIBRARIES += performance/libnanox-opencl-api.la performance/libnanox-pe-opencl.la

nanos_perfo_cxxflags=$(common_includes) $(common_perfo_cxxflags) $(opencl_includes)

performance_libopencl_la_CXXFLAGS=$(nanos_perfo_cxxflags) $(opencl_api_flags)
performance_libopencl_la_LDFLAGS=$(opencl_ldflags)
performance_libopencl_la_SOURCES=$(opencl_sources)

performance_libnanox_opencl_api_la_CXXFLAGS=$(nanos_perfo_cxxflags) $(opencl_api_flags)
performance_libnanox_opencl_api_la_LDFLAGS=$(version_flags)
performance_libnanox_opencl_api_la_SOURCES=$(opencl_api_sources)

performance_libnanox_pe_opencl_la_CXXFLAGS=$(nanos_perfo_cxxflags)
performance_libnanox_pe_opencl_la_LDFLAGS=$(ld_plugin_flags) $(opencl_ldflags)
performance_libnanox_pe_opencl_la_SOURCES=$(pe_opencl_sources)
endif
