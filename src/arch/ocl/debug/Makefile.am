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

include $(top_srcdir)/src/arch/ocl/version_commons.am
include $(top_srcdir)/src/common.am

if DEBUG_VERSION
noinst_LTLIBRARIES = libocl.la
debug_LTLIBRARIES = libnanox-pe-ocl.la

nanos_debug_cxxflags=$(common_includes) $(common_debug_cxxflags) $(ocl_includes)

libocl_la_CXXFLAGS=$(nanos_debug_cxxflags)
libocl_la_LDFLAGS=$(ocl_ldflags)
libocl_la_SOURCES=$(ocl_sources)

libnanox_pe_ocl_la_CXXFLAGS=$(nanos_debug_cxxflags)
libnanox_pe_ocl_la_LDFLAGS=$(ld_plugin_flags) $(ocl_ldflags)
libnanox_pe_ocl_la_SOURCES=$(pe_ocl_sources)
endif
