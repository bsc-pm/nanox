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

lib_LTLIBRARIES=libnanox-c.la

include_HEADERS=\
	nanos.h
	$(END)

sources=\
	nanos.h\
	nanos_wd.cpp\
	nanos_sync.cpp\
	nanos_team.cpp\
	nanos_err.cpp\
	nanos_sys.cpp\
	$(END)

libnanox_c_la_CXXFLAGS=$(common_includes) $(common_cxxflags) $(performance_cflags)
libnanox_c_la_LDFLAGS=$(version_flags)
libnanox_c_la_LIBADD=$(top_builddir)/src/core/libnanox.la
libnanox_c_la_SOURCES=$(sources)
libnanox_c_la_DEPENDENCIES=$(top_builddir)/src/core/libnanox.la

