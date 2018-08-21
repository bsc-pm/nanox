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

EXTRA_DIST= \
	$(END)

pkginclude_HEADERS=\
	nanos-fpga.h \
	$(END)

devinclude_HEADERS = \
    fpgaconfig.hpp \
    fpgadd.hpp \
    fpgadevice.hpp \
    fpgaprocessor.hpp \
    fpgaprocessorinfo.hpp \
    fpgathread.hpp \
    fpgamemorytransfer.hpp \
    fpgaworker.hpp \
    fpgapinnedallocator.hpp \
    $(END) 

fpga_api_sources = \
    nanos-fpga.h \
    fpga-api.cpp \
    $(END)

fpga_sources = \
    fpgaconfig.hpp \
    fpgaconfig.cpp \
    fpgadd.hpp \
    fpgadd.cpp \
    fpgadevice.hpp \
    fpgadevice.cpp \
    fpgaprocessor.hpp \
    fpgaprocessor.cpp \
    fpgaprocessorinfo.hpp \
    fpgathread.hpp \
    fpgathread.cpp \
    fpgamemorytransfer.hpp \
    fpgamemorytransfer.cpp \
    fpgaworker.hpp \
    fpgaworker.cpp \
    fpgapinnedallocator.hpp \
    fpgapinnedallocator.cpp \
    $(END)

pe_fpga_sources = \
    fpgaplugin.cpp \
    $(END)

noinst_LTLIBRARIES =
debug_LTLIBRARIES =
instrumentation_LTLIBRARIES =
instrumentation_debug_LTLIBRARIES =
performance_LTLIBRARIES =

if is_debug_enabled
noinst_LTLIBRARIES += debug/libfpga.la
debug_LTLIBRARIES += debug/libnanox-fpga-api.la debug/libnanox-pe-fpga.la

debug_libfpga_la_CPPFLAGS=$(common_debug_CPPFLAGS) @xdmainc@
debug_libfpga_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libfpga_la_LDFLAGS=$(AM_LDFLAGS) @xdmalib@
debug_libfpga_la_SOURCES=$(fpga_sources)

debug_libnanox_fpga_api_la_CPPFLAGS=$(common_debug_CPPFLAGS) $(api_includes) @xdmainc@
debug_libnanox_fpga_api_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_fpga_api_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
debug_libnanox_fpga_api_la_SOURCES=$(fpga_api_sources)

debug_libnanox_pe_fpga_la_CPPFLAGS=$(common_debug_CPPFLAGS) @xdmainc@
debug_libnanox_pe_fpga_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_pe_fpga_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @xdmalib@
debug_libnanox_pe_fpga_la_SOURCES=$(pe_fpga_sources)
endif

if is_instrumentation_debug_enabled
noinst_LTLIBRARIES += instrumentation-debug/libfpga.la
instrumentation_debug_LTLIBRARIES += instrumentation_debug/libnanox-fpga-api.la instrumentation_debug/libnanox-pe-fpga.la

instrumentation_debug_libfpga_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) @xdmainc@
instrumentation_debug_libfpga_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libfpga_la_LDFLAGS=$(AM_LDFLAGS) @xdmalib@
instrumentation_debug_libfpga_la_SOURCES=$(fpga_sources)

instrumentation_debug_libnanox_fpga_api_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) $(api_includes) @xdmainc@
instrumentation_debug_libnanox_fpga_api_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_fpga_api_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
instrumentation_debug_libnanox_fpga_api_la_SOURCES=$(fpga_api_sources)

instrumentation_debug_libnanox_pe_fpga_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) @xdmainc@
instrumentation_debug_libnanox_pe_fpga_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_pe_fpga_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @xdmalib@
instrumentation_debug_libnanox_pe_fpga_la_SOURCES=$(pe_fpga_sources)
endif

if is_instrumentation_enabled
noinst_LTLIBRARIES += instrumentation/libfpga.la
instrumentation_LTLIBRARIES += instrumentation/libnanox-fpga-api.la instrumentation/libnanox-pe-fpga.la

instrumentation_libfpga_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) @xdmainc@
instrumentation_libfpga_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libfpga_la_LDFLAGS=$(AM_LDFLAGS) @xdmalib@
instrumentation_libfpga_la_SOURCES=$(fpga_sources)

instrumentation_libnanox_fpga_api_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) $(api_includes) @xdmainc@
instrumentation_libnanox_fpga_api_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_fpga_api_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
instrumentation_libnanox_fpga_api_la_SOURCES=$(fpga_api_sources)

instrumentation_libnanox_pe_fpga_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) @xdmainc@
instrumentation_libnanox_pe_fpga_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_pe_fpga_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @xdmalib@
instrumentation_libnanox_pe_fpga_la_SOURCES=$(pe_fpga_sources)
endif

if is_performance_enabled
noinst_LTLIBRARIES += performance/libfpga.la
performance_LTLIBRARIES += performance/libnanox-fpga-api.la performance/libnanox-pe-fpga.la

performance_libfpga_la_CPPFLAGS=$(common_performance_CPPFLAGS) @xdmainc@
performance_libfpga_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libfpga_la_LDFLAGS=$(AM_LDFLAGS) @xdmalib@
performance_libfpga_la_SOURCES=$(fpga_sources)

performance_libnanox_fpga_api_la_CPPFLAGS=$(common_performance_CPPFLAGS) $(api_includes) @xdmainc@
performance_libnanox_fpga_api_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_fpga_api_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
performance_libnanox_fpga_api_la_SOURCES=$(fpga_api_sources)

performance_libnanox_pe_fpga_la_CPPFLAGS=$(common_performance_CPPFLAGS) @xdmainc@
performance_libnanox_pe_fpga_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_pe_fpga_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @xdmalib@
performance_libnanox_pe_fpga_la_SOURCES=$(pe_fpga_sources)
endif

