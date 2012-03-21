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

include $(top_srcdir)/src/plugins/worksharing/version_commons.am
include $(top_srcdir)/src/common.am

if INSTRUMENTATION_DEBUG_VERSION
instrdebug_LTLIBRARIES = \
	libnanox-worksharing-static_for.la \
	libnanox-worksharing-dynamic_for.la \
	libnanox-worksharing-guided_for.la \
	$(END)

nanos_instrdebug_cxxflags=$(common_includes) $(common_instrdebug_cxxflags)

libnanox_worksharing_static_for_la_CXXFLAGS = $(nanos_instrdebug_cxxflags)
libnanox_worksharing_static_for_la_LDFLAGS = $(ld_plugin_flags)
libnanox_worksharing_static_for_la_SOURCES = $(worksharing_static_for_sources)

libnanox_worksharing_dynamic_for_la_CXXFLAGS = $(nanos_instrdebug_cxxflags)
libnanox_worksharing_dynamic_for_la_LDFLAGS = $(ld_plugin_flags)
libnanox_worksharing_dynamic_for_la_SOURCES = $(worksharing_dynamic_for_sources)

libnanox_worksharing_guided_for_la_CXXFLAGS = $(nanos_instrdebug_cxxflags)
libnanox_worksharing_guided_for_la_LDFLAGS = $(ld_plugin_flags)
libnanox_worksharing_guided_for_la_SOURCES = $(worksharing_guided_for_sources)

endif

