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

lib_LTLIBRARIES=libnanox.la

include_HEADERS=nanos-int.h

if PERFORMANCE
  lib_LTLIBRARIES += libnanox-pfm.la
endif

# Common flags and files
nanos_cxxflags=$(common_includes) $(common_cxxflags)
nanos_ldflags=$(version_flags) @NANOS_CONFIG_LIBDIRS@
nanos_libadd=\
	-lpthread \
	-ldl \
	../arch/@OS@/libos.la \
	../support/libsupport.la \
	@NANOS_CONFIG_LIBS@\
	$(END)

nanos_sources=\
	nanos-int.h \
	barrier_fwd.hpp \
	barrier.hpp \
	threadteam_fwd.hpp \
	threadteam.hpp \
	threadteam.cpp \
	basethread_fwd.hpp\
	basethread.hpp\
	basethread.cpp\
	processingelement_fwd.hpp \
	processingelement.hpp \
	processingelement.cpp \
	accelerator_fwd.hpp \
	accelerator.hpp \
	accelerator.cpp \
	schedule_fwd.hpp \
	schedule.hpp \
	schedule.cpp \
	system_fwd.hpp \
	system_decl.hpp\
	system.hpp \
	system.cpp \
	wddeque_fwd.hpp \
	wddeque.hpp \
	workgroup_fwd.hpp \
	workgroup.hpp \
	workgroup.cpp \
	workdescriptor_fwd.hpp \
	workdescriptor_decl.hpp \
	workdescriptor.hpp \
	workdescriptor.cpp \
	slicer_fwd.hpp \
	slicer.hpp \
	instrumentorcontext_decl.hpp \
	instrumentorcontext.hpp \
	instrumentor_fwd.hpp \
	instrumentor_decl.hpp \
	instrumentor.hpp \
	instrumentor.cpp \
	throttle_fwd.hpp \
	throttle.hpp \
	dependency_fwd.hpp \
	dependency.hpp \
	copydata_fwd.hpp \
	copydata.hpp \
	dependableobject_fwd.hpp \
	dependableobject.hpp \
	dependableobjectwd.hpp \
	dependableobjectwd_fwd.hpp \
	dependableobjectwd.cpp \
	trackableobject_fwd.hpp \
	trackableobject.hpp \
	dependenciesdomain_fwd.hpp \
	dependenciesdomain.hpp \
	dependenciesdomain.cpp \
	synchronizedcondition_fwd.hpp \
	synchronizedcondition_decl.hpp \
	synchronizedcondition.hpp \
	$(END)

# Default
libnanox_la_CFLAGS=$(nanos_cflags)
libnanox_la_CXXFLAGS=$(nanos_cxxflags)
libnanox_la_LDFLAGS=$(nanos_ldflags) 
libnanox_la_LIBADD=$(nanos_libadd)
libnanox_la_SOURCES=$(nanos_sources)
libnanox_la_DEPENDENCIES=$(nanos_libadd)

# Performance
libnanox_pfm_la_CFLAGS=$(nanos_cflags) $(performance_cflags)
libnanox_pfm_la_CXXFLAGS=$(nanos_cxxflags) $(performance_cflags)
libnanox_pfm_la_LDFLAGS=$(nanos_ldflags) $(nanos_libadd)
libnanox_pfm_la_LIBADD=$(nanos_libadd)
libnanox_pfm_la_SOURCES=$(nanos_sources)
libnanox_pfm_la_DEPENDENCIES=$(nanos_libadd)
