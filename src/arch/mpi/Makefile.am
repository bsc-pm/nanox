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

SUBDIRS = performance debug instrumentation instrumentation-debug

include $(top_srcdir)/src/common.am

pkginclude_HEADERS=\
	nanos-mpi.h \
	$(END)
	
	
devinclude_HEADERS = \
	mpiprocessor_fwd.hpp \
	mpiprocessor_decl.hpp \
	mpiprocessor.hpp \
	mpiremotenode_fwd.hpp \
	mpiremotenode_decl.hpp \
	mpiremotenode.hpp \
	mpidevice_decl.hpp \
	mpidevice.hpp \
	mpidd.hpp \
	mpithread.hpp \
	$(END)

mpi_api_sources = \
	nanos-mpi.h \
	mpi-api.cpp \
	$(END)

mpi_sources = \
	mpiprocessor_fwd.hpp \
	mpiprocessor_decl.hpp \
	mpiprocessor.hpp \
	mpiprocessor.cpp \
	mpiremotenode_fwd.hpp \
	mpiremotenode_decl.hpp \
	mpiremotenode.hpp \
	mpiremotenode.cpp \
	mpidevice_decl.hpp \
	mpidevice.hpp \
	mpidevice.cpp \
	mpidd.hpp \
	mpidd.cpp \
	mpithread.hpp \
	mpithread.cpp \
	$(END)

pe_mpi_sources = \
	mpiplugin.cpp \
	$(END)

mpi_api_flags = -I$(top_srcdir)/src/apis/c \
                -DNANOX_PREFIX=\"$(NANOX_MPI_PREFIX)\"

noinst_LTLIBRARIES =
debug_LTLIBRARIES =
instrumentation_LTLIBRARIES =
instrumentation_debug_LTLIBRARIES =
performance_LTLIBRARIES =


if PERFORMANCE_VERSION
noinst_LTLIBRARIES += libmpi.la
performance_LTLIBRARIES += libnanox-mpi-api.la
performance_LTLIBRARIES += libnanox-pe-mpi.la

nanos_perfo_cxxflags=$(common_includes) $(common_perfo_cxxflags) $(mpi_ult_flags)

libmpi_la_CXXFLAGS=$(nanos_perfo_cxxflags) $(mpi_api_flags)
libmpi_la_SOURCES=$(mpi_sources) $(mpi_api_sources)

libnanox_mpi_api_la_CXXFLAGS=$(nanos_perfo_cxxflags) $(mpi_api_flags)
libnanox_mpi_api_la_LDFLAGS=$(version_flags)
libnanox_mpi_api_la_SOURCES=$(mpi_api_sources)

libnanox_pe_mpi_la_CXXFLAGS=$(nanos_perfo_cxxflags) $(mpi_api_flags)
libnanox_pe_mpi_la_LDFLAGS=$(ld_plugin_flags)
libnanox_pe_mpi_la_LIBADD=$(common_libadd)
libnanox_pe_mpi_la_SOURCES=$(pe_mpi_sources)
endif



if DEBUG_VERSION
noinst_LTLIBRARIES += libmpi.la
debug_LTLIBRARIES += libnanox-mpi-api.la
debug_LTLIBRARIES += libnanox-pe-mpi.la


nanos_debug_cxxflags=$(common_includes) $(common_debug_cxxflags) $(mpi_ult_flags)

libmpi_la_CXXFLAGS=$(nanos_debug_cxxflags)  $(mpi_api_flags)
libmpi_la_SOURCES=$(mpi_sources) $(mpi_api_sources)

libnanox_mpi_api_la_CXXFLAGS=$(nanos_debug_cxxflags) $(mpi_api_flags)
libnanox_mpi_api_la_LDFLAGS=$(version_flags)
libnanox_mpi_api_la_SOURCES=$(mpi_api_sources)

libnanox_pe_mpi_la_CXXFLAGS=$(nanos_debug_cxxflags) $(mpi_api_flags)
libnanox_pe_mpi_la_LDFLAGS=$(ld_plugin_flags)
libnanox_pe_mpi_la_LIBADD=$(common_libadd)
libnanox_pe_mpi_la_SOURCES=$(pe_mpi_sources)
endif

if INSTRUMENTATION_VERSION
noinst_LTLIBRARIES += libmpi.la
instrumentation_LTLIBRARIES += libnanox-mpi-api.la
instrumentation_LTLIBRARIES += libnanox-pe-mpi.la


nanos_instr_cxxflags=$(common_includes) $(common_instr_cxxflags) $(mpi_ult_flags)

libmpi_la_CXXFLAGS=$(nanos_instr_cxxflags)  $(mpi_api_flags)
libmpi_la_SOURCES=$(mpi_sources) $(mpi_api_sources)

libnanox_mpi_api_la_CXXFLAGS=$(nanos_instr_cxxflags) $(mpi_api_flags)
libnanox_mpi_api_la_LDFLAGS=$(version_flags)
libnanox_mpi_api_la_SOURCES=$(mpi_api_sources)

libnanox_pe_mpi_la_CXXFLAGS=$(nanos_instr_cxxflags) $(mpi_api_flags)
libnanox_pe_mpi_la_LDFLAGS=$(ld_plugin_flags)
libnanox_pe_mpi_la_LIBADD=$(common_libadd)
libnanox_pe_mpi_la_SOURCES=$(pe_mpi_sources)
endif

if INSTRUMENTATION_DEBUG_VERSION
noinst_LTLIBRARIES += libmpi.la
instrumentation_debug_LTLIBRARIES += libnanox-mpi-api.la
instrumentation_debug_LTLIBRARIES += libnanox-pe-mpi.la


nanos_instrdebug_cxxflags=$(common_includes) $(common_instrdebug_cxxflags) $(mpi_ult_flags)

libmpi_la_CXXFLAGS=$(nanos_instrdebug_cxxflags)  $(mpi_api_flags)
libmpi_la_SOURCES=$(mpi_sources) $(mpi_api_sources)

libnanox_mpi_api_la_CXXFLAGS=$(nanos_instrdebug_cxxflags) $(mpi_api_flags)
libnanox_mpi_api_la_LDFLAGS=$(version_flags)
libnanox_mpi_api_la_SOURCES=$(mpi_api_sources)

libnanox_pe_mpi_la_CXXFLAGS=$(nanos_instrdebug_cxxflags) $(mpi_api_flags)
libnanox_pe_mpi_la_LDFLAGS=$(ld_plugin_flags)
libnanox_pe_mpi_la_LIBADD=$(common_libadd)
libnanox_pe_mpi_la_SOURCES=$(pe_mpi_sources)
endif
