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
	$(END)

pe_fpga_sources = \
    fpgaplugin.cpp \
	$(END)

fpga_api_flags = -I$(top_srcdir)/src/apis/c

#fpga_ldflags = -L@XDMADIR@/lib/ -lxlnk
fpga_ldflags = -L@XDMADIR@/lib/ -lxdma-static
fpga_includes = -I@XDMADIR@/include/

noinst_LTLIBRARIES =
debug_LTLIBRARIES =
instrumentation_LTLIBRARIES =
instrumentation_debug_LTLIBRARIES =
performance_LTLIBRARIES =

if DEBUG_VERSION
noinst_LTLIBRARIES += debug/libfpga.la
debug_LTLIBRARIES += debug/libnanox-fpga-api.la debug/libnanox-pe-fpga.la

nanos_debug_cxxflags=$(common_includes) $(common_debug_cxxflags) $(fpga_includes)

debug_libfpga_la_CFLAGS=$(fpga_includes)
debug_libfpga_la_CXXFLAGS=$(nanos_debug_cxxflags)
debug_libfpga_la_LDFLAGS=$(fpga_ldflags)
debug_libfpga_la_SOURCES=$(fpga_sources)

debug_libnanox_fpga_api_la_CXXFLAGS=$(nanos_debug_cxxflags) $(fpga_api_flags)
debug_libnanox_fpga_api_la_LDFLAGS=$(version_flags)
debug_libnanox_fpga_api_la_SOURCES=$(fpga_api_sources)

debug_libnanox_pe_fpga_la_CXXFLAGS=$(nanos_debug_cxxflags)
debug_libnanox_pe_fpga_la_LDFLAGS=$(ld_plugin_flags) $(fpga_ldflags)
debug_libnanox_pe_fpga_la_SOURCES=$(pe_fpga_sources)
endif

if INSTRUMENTATION_VERSION
noinst_LTLIBRARIES += instrumentation/libfpga.la
instrumentation_LTLIBRARIES += instrumentation/libnanox-fpga-api.la instrumentation/libnanox-pe-fpga.la

nanos_instr_cxxflags=$(common_includes) $(common_instr_cxxflags) $(fpga_includes)

instrumentation_libfpga_la_CFLAGS=$(fpga_includes)
instrumentation_libfpga_la_CXXFLAGS=$(nanos_instr_cxxflags)
instrumentation_libfpga_la_LDFLAGS=$(fpga_ldflags)
instrumentation_libfpga_la_SOURCES=$(fpga_sources)

instrumentation_libnanox_fpga_api_la_CXXFLAGS=$(nanos_instr_cxxflags) $(fpga_api_flags)
instrumentation_libnanox_fpga_api_la_LDFLAGS=$(version_flags)
instrumentation_libnanox_fpga_api_la_SOURCES=$(fpga_api_sources)

instrumentation_libnanox_pe_fpga_la_CXXFLAGS=$(nanos_instr_cxxflags)
instrumentation_libnanox_pe_fpga_la_LDFLAGS=$(ld_plugin_flags) $(fpga_ldflags)
instrumentation_libnanox_pe_fpga_la_SOURCES=$(pe_fpga_sources)
endif

if INSTRUMENTATION_DEBUG_VERSION
noinst_LTLIBRARIES += instrumentation-debug/libfpga.la
instrumentation_debug_LTLIBRARIES += instrumentation-debug/libnanox-fpga-api.la instrumentation-debug/libnanox-pe-fpga.la

nanos_instrdebug_cxxflags=$(common_includes) $(common_instrdebug_cxxflags) $(fpga_includes)

instrumentation_debug_libfpga_la_CFLAGS=$(fpga_includes)
instrumentation_debug_libfpga_la_CXXFLAGS=$(nanos_instrdebug_cxxflags)
instrumentation_debug_libfpga_la_LDFLAGS=$(fpga_ldflags)
instrumentation_debug_libfpga_la_SOURCES=$(fpga_sources)

instrumentation_debug_libnanox_fpga_api_la_CXXFLAGS=$(nanos_instrdebug_cxxflags) $(fpga_api_flags)
instrumentation_debug_libnanox_fpga_api_la_LDFLAGS=$(version_flags)
instrumentation_debug_libnanox_fpga_api_la_SOURCES=$(fpga_api_sources)

instrumentation_debug_libnanox_pe_fpga_la_CXXFLAGS=$(nanos_instrdebug_cxxflags)
instrumentation_debug_libnanox_pe_fpga_la_LDFLAGS=$(ld_plugin_flags) $(fpga_ldflags)
instrumentation_debug_libnanox_pe_fpga_la_SOURCES=$(pe_fpga_sources)
endif

if PERFORMANCE_VERSION
noinst_LTLIBRARIES += performance/libfpga.la
performance_LTLIBRARIES += performance/libnanox-fpga-api.la performance/libnanox-pe-fpga.la

nanos_perfo_cxxflags=$(common_includes) $(common_perfo_cxxflags) $(fpga_includes)

performance_libfpga_la_CFLAGS=$(fpga_includes)
performance_libfpga_la_CXXFLAGS=$(nanos_perfo_cxxflags)
performance_libfpga_la_LDFLAGS=$(fpga_ldflags)
performance_libfpga_la_SOURCES=$(fpga_sources)

performance_libnanox_fpga_api_la_CXXFLAGS=$(nanos_perfo_cxxflags) $(fpga_api_flags)
performance_libnanox_fpga_api_la_LDFLAGS=$(version_flags)
performance_libnanox_fpga_api_la_SOURCES=$(fpga_api_sources)

performance_libnanox_pe_fpga_la_CXXFLAGS=$(nanos_perfo_cxxflags)
performance_libnanox_pe_fpga_la_LDFLAGS=$(ld_plugin_flags) $(fpga_ldflags)
performance_libnanox_pe_fpga_la_SOURCES=$(pe_fpga_sources)
endif

