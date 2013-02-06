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

#TODO: Fix instrumentation and debug makefiles so they are named opencl instead of opencl and they can use headers in this folder
include $(top_srcdir)/src/arch/opencl/version_commons.am
include $(top_srcdir)/src/common.am

if DEBUG_VERSION
noinst_LTLIBRARIES = libopencl.la
debug_LTLIBRARIES = libnanox-opencl-api.la libnanox-pe-opencl.la

nanos_debug_cxxflags=$(common_includes) $(common_debug_cxxflags) $(opencl_includes)

libopencl_la_CXXFLAGS=$(nanos_debug_cxxflags) $(opencl_api_flags)
libopencl_la_LDFLAGS=$(opencl_ldflags)
libopencl_la_SOURCES=$(opencl_sources)

libnanox_opencl_api_la_CXXFLAGS=$(nanos_debug_cxxflags) $(opencl_api_flags)
libnanox_opencl_api_la_LDFLAGS=$(version_flags)
libnanox_opencl_api_la_SOURCES=$(opencl_api_sources)

libnanox_pe_opencl_la_CXXFLAGS=$(nanos_debug_cxxflags)
libnanox_pe_opencl_la_LDFLAGS=$(ld_plugin_flags) $(opencl_ldflags)
libnanox_pe_opencl_la_SOURCES=$(pe_opencl_sources)
endif
