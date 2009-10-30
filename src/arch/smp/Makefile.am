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

noinst_LTLIBRARIES = libsmp.la
lib_LTLIBRARIES = libnanox-pe-smp.la

libsmp_la_CXXFLAGS=$(common_includes) $(common_cxxflags) 
libsmp_la_SOURCES = \
		smpdd.hpp \
		smpdd.cpp \
		@SMP_ARCH@/stack.cpp \
		@SMP_ARCH@/stack.s \
		$(END)


libnanox_pe_smp_la_CXXFLAGS=$(common_includes) $(common_cxxflags) 
libnanox_pe_smp_la_LDFLAGS = $(ld_plugin_flags)
libnanox_pe_smp_la_SOURCES = \
		smpplugin.cpp \
		smpprocessor.hpp \
		smpprocessor.cpp \
		smpthread.hpp \
		smpthread.cpp \
		$(END)

