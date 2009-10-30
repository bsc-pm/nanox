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
	libnanox-cutoff-tasknum.la \
	libnanox-cutoff-idle.la \
	libnanox-cutoff-level.la \
	libnanox-cutoff-ready.la \
	$(END)

noinst_LTLIBRARIES = libcutoff.la

libcutoff_la_CXXFLAGS= $(common_includes) $(common_cxxflags)
libcutoff_la_SOURCES = \
	dummy_cutoff.cpp \
	$(END)

libnanox_cutoff_tasknum_la_CXXFLAGS = $(common_includes) $(common_cxxflags)
libnanox_cutoff_tasknum_la_LDFLAGS = $(ld_plugin_flags)
libnanox_cutoff_tasknum_la_SOURCES = \
        tasknum_cutoff.cpp \
        $(END)

libnanox_cutoff_idle_la_CXXFLAGS = $(common_includes) $(common_cxxflags)
libnanox_cutoff_idle_la_LDFLAGS = $(ld_plugin_flags)
libnanox_cutoff_idle_la_SOURCES = \
        idle_cutoff.cpp \
        $(END)

libnanox_cutoff_level_la_CXXFLAGS = $(common_includes) $(common_cxxflags)
libnanox_cutoff_level_la_LDFLAGS = $(ld_plugin_flags)
libnanox_cutoff_level_la_SOURCES = \
        level_cutoff.cpp \
        $(END)

libnanox_cutoff_ready_la_CXXFLAGS = $(common_includes) $(common_cxxflags)
libnanox_cutoff_ready_la_LDFLAGS = $(ld_plugin_flags)
libnanox_cutoff_ready_la_SOURCES = \
        ready_cutoff.cpp \
        $(END)