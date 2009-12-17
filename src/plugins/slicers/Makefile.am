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

include $(top_srcdir)/src/common.am

lib_LTLIBRARIES = \
	libnanox-slicer-static_for.la \
	libnanox-slicer-dynamic_for.la \
	libnanox-slicer-guided_for.la \
	libnanox-slicer-repeat_n.la \
	$(END)

libnanox_slicer_static_for_la_CXXFLAGS = $(common_includes) $(common_cxxflags)
libnanox_slicer_static_for_la_LDFLAGS = $(ld_plugin_flags)
libnanox_slicer_static_for_la_SOURCES = \
	static_for.cpp \
	$(END)

libnanox_slicer_dynamic_for_la_CXXFLAGS = $(common_includes) $(common_cxxflags)
libnanox_slicer_dynamic_for_la_LDFLAGS = $(ld_plugin_flags)
libnanox_slicer_dynamic_for_la_SOURCES = \
	dynamic_for.cpp \
	$(END)

libnanox_slicer_guided_for_la_CXXFLAGS = $(common_includes) $(common_cxxflags)
libnanox_slicer_guided_for_la_LDFLAGS = $(ld_plugin_flags)
libnanox_slicer_guided_for_la_SOURCES = \
	guided_for.cpp \
	$(END)

libnanox_slicer_repeat_n_la_CXXFLAGS = $(common_includes) $(common_cxxflags)
libnanox_slicer_repeat_n_la_LDFLAGS = $(ld_plugin_flags)
libnanox_slicer_repeat_n_la_SOURCES = \
	repeat_n.cpp \
	$(END)

