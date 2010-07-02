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
	libnanox-instrumentor-empty_trace.la \
	libnanox-instrumentor-print_trace.la \
	$(END)

if INSTRUMENTATION_EXTRAE
mpitrace_cxxflags=-I@MPITRACE_INC@ -DMPITRACE_BIN='"@MPITRACE_BIN@"'
mpitrace_ldflags=-L@MPITRACE_LIB@ -lnanostrace

lib_LTLIBRARIES += libnanox-instrumentor-paraver.la
endif

libnanox_instrumentor_empty_trace_la_CXXFLAGS = $(common_includes) $(common_cxxflags)
libnanox_instrumentor_empty_trace_la_LDFLAGS = $(ld_plugin_flags)
libnanox_instrumentor_empty_trace_la_SOURCES = \
	empty_trace.cpp \
	$(END)

libnanox_instrumentor_print_trace_la_CXXFLAGS = $(common_includes) $(common_cxxflags)
libnanox_instrumentor_print_trace_la_LDFLAGS = $(ld_plugin_flags)
libnanox_instrumentor_print_trace_la_SOURCES = \
	print_trace.cpp \
	$(END)

libnanox_instrumentor_paraver_la_CXXFLAGS = $(common_includes) $(common_cxxflags) $(mpitrace_cxxflags) 
libnanox_instrumentor_paraver_la_LDFLAGS = $(ld_plugin_flags) $(mpitrace_ldflags)
libnanox_instrumentor_paraver_la_SOURCES = \
	paraver.cpp \
        ompi_services.cpp \
	$(END)
