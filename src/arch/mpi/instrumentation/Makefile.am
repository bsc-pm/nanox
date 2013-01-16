#####################################################################################
#      Copyright 2012 Barcelona Supercomputing Center                               #
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

include $(top_srcdir)/src/arch/mpi/version_commons.am
include $(top_srcdir)/src/common.am

if INSTRUMENTATION_VERSION
noinst_LTLIBRARIES = libmpi.la
instrumentation_LTLIBRARIES = libnanox-mpi-api.la
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
