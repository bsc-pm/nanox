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

pkginclude_HEADERS=\
	nanos-opencl.h \
	$(END)
	
EXTRA_DIST = \
	$(END)

devinclude_HEADERS = \
    openclconfig.hpp \
    opencldd.hpp \
    opencldevice_decl.hpp \
    openclprocessor_fwd.hpp \
    opencldevice.hpp \
    openclprocessor_decl.hpp \
    openclprocessor.hpp \
    openclcache.hpp \
    openclthread_decl.hpp \
    openclthread.hpp \
    openclutils.hpp \
    sharedmemallocator.hpp \
    openclevent.hpp \
    openclevent_decl.hpp \
    openclprofiler.hpp \
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
    openclprofiler.cpp \
    $(END)

pe_opencl_sources = \
	openclplugin.hpp \
	openclplugin.cpp \
	$(END)

noinst_LTLIBRARIES =
debug_LTLIBRARIES =
instrumentation_LTLIBRARIES =
instrumentation_debug_LTLIBRARIES =
performance_LTLIBRARIES =

#TODO: Fix instrumentation and debug makefiles so they are named opencl instead of opencl and they can use headers in this folder

if is_debug_enabled
noinst_LTLIBRARIES += debug/libopencl.la
debug_LTLIBRARIES += debug/libnanox-opencl-api.la debug/libnanox-pe-opencl.la

debug_libopencl_la_CPPFLAGS=$(common_debug_CPPFLAGS) @openclinc@
debug_libopencl_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libopencl_la_LDFLAGS=$(AM_LDFLAGS) @opencllib@
debug_libopencl_la_LIBADD=@opencllibs@
debug_libopencl_la_SOURCES=$(opencl_sources)

debug_libnanox_opencl_api_la_CPPFLAGS=$(common_debug_CPPFLAGS) $(api_includes) @openclinc@
debug_libnanox_opencl_api_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_opencl_api_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags) @opencllib@
debug_libnanox_opencl_api_la_LIBADD=@opencllibs@
debug_libnanox_opencl_api_la_SOURCES=$(opencl_api_sources)

debug_libnanox_pe_opencl_la_CPPFLAGS=$(common_debug_CPPFLAGS) @openclinc@
debug_libnanox_pe_opencl_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_pe_opencl_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @opencllib@
debug_libnanox_pe_opencl_la_LIBADD=@opencllibs@
debug_libnanox_pe_opencl_la_SOURCES=$(pe_opencl_sources)
endif

if is_instrumentation_debug_enabled
noinst_LTLIBRARIES += instrumentation-debug/libopencl.la
instrumentation_debug_LTLIBRARIES += instrumentation-debug/libnanox-opencl-api.la instrumentation-debug/libnanox-pe-opencl.la

instrumentation_debug_libopencl_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) @openclinc@
instrumentation_debug_libopencl_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libopencl_la_LDFLAGS=$(AM_LDFLAGS) @opencllib@
instrumentation_debug_libopencl_la_LIBADD=@opencllibs@
instrumentation_debug_libopencl_la_SOURCES=$(opencl_sources)

instrumentation_debug_libnanox_opencl_api_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) $(api_includes) @openclinc@
instrumentation_debug_libnanox_opencl_api_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_opencl_api_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags) @opencllib@
instrumentation_debug_libnanox_opencl_api_la_LIBADD=@opencllibs@
instrumentation_debug_libnanox_opencl_api_la_SOURCES=$(opencl_api_sources)

instrumentation_debug_libnanox_pe_opencl_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) @openclinc@
instrumentation_debug_libnanox_pe_opencl_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_pe_opencl_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @opencllib@
instrumentation_debug_libnanox_pe_opencl_la_LIBADD=@opencllibs@
instrumentation_debug_libnanox_pe_opencl_la_SOURCES=$(pe_opencl_sources)
endif

if is_instrumentation_enabled
noinst_LTLIBRARIES += instrumentation/libopencl.la
instrumentation_LTLIBRARIES += instrumentation/libnanox-opencl-api.la instrumentation/libnanox-pe-opencl.la

instrumentation_libopencl_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) @openclinc@
instrumentation_libopencl_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libopencl_la_LDFLAGS=$(AM_LDFLAGS) @opencllib@
instrumentation_libopencl_la_LIBADD=@opencllibs@
instrumentation_libopencl_la_SOURCES=$(opencl_sources)

instrumentation_libnanox_opencl_api_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) $(api_includes) @openclinc@
instrumentation_libnanox_opencl_api_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_opencl_api_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags) @opencllib@
instrumentation_libnanox_opencl_api_la_LIBADD=@opencllibs@
instrumentation_libnanox_opencl_api_la_SOURCES=$(opencl_api_sources)

instrumentation_libnanox_pe_opencl_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) @openclinc@
instrumentation_libnanox_pe_opencl_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_pe_opencl_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @opencllib@
instrumentation_libnanox_pe_opencl_la_LIBADD=@opencllibs@
instrumentation_libnanox_pe_opencl_la_SOURCES=$(pe_opencl_sources)
endif

if is_performance_enabled
noinst_LTLIBRARIES += performance/libopencl.la
performance_LTLIBRARIES += performance/libnanox-opencl-api.la performance/libnanox-pe-opencl.la

performance_libopencl_la_CPPFLAGS=$(common_performance_CPPFLAGS) @openclinc@
performance_libopencl_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libopencl_la_LDFLAGS=$(AM_LDFLAGS) @opencllib@
performance_libopencl_la_LIBADD=@opencllibs@
performance_libopencl_la_SOURCES=$(opencl_sources)

performance_libnanox_opencl_api_la_CPPFLAGS=$(common_performance_CPPFLAGS) $(api_includes) @openclinc@
performance_libnanox_opencl_api_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_opencl_api_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags) @opencllib@
performance_libnanox_opencl_api_la_LIBADD=@opencllibs@
performance_libnanox_opencl_api_la_SOURCES=$(opencl_api_sources)

performance_libnanox_pe_opencl_la_CPPFLAGS=$(common_performance_CPPFLAGS) @openclinc@
performance_libnanox_pe_opencl_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_pe_opencl_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @opencllib@
performance_libnanox_pe_opencl_la_LIBADD=@opencllibs@
performance_libnanox_pe_opencl_la_SOURCES=$(pe_opencl_sources)
endif

