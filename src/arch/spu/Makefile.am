#####################################################################################
#      Copyright 2009-2018 Barcelona Supercomputing Center                          #
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
#      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            #
#####################################################################################

include $(top_srcdir)/src/common.am

EXTRAD_DIST= spu_idle.spu

SPUCCFLAGS=-I @CELLSDK_INC@

.spu.o: $<
	$(SPUCC) $(SPUCCFLAGS) -x c -o $@ $<

.libs/libspu_embed.a : spu_idle.o
	$(PPUEMBED) -m32 spu_idle $< null.o
	$(AR) cru $@ null.o
	$(RM) -rf null.o
	$(RANLIB) $@

devinclude_HEADERS = \
	$(END) 

spu_sources = \
	spudd.hpp \
	spudd.cpp \
	spuprocessor.hpp \
	spuprocessor.cpp \
	sputhread.hpp \
	sputhread.cpp \
	$(END)

spu_libadd = -lspu_embed
spu_ccflags = -I @CELLSDK_INC@

noinst_LTLIBRARIES =

if is_debug_enabled
noinst_LTLIBRARIES += debug/libspu.la


debug_libspu_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libspu_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libspu_la_SOURCES=$(spu_sources)
debug_libspu_la_DEPENDENCIES=.libs/libspu_embed.a
debug_libspu_la_LIBADD=$(spu_libadd)
endif

if is_instrumentation_enabled
noinst_LTLIBRARIES += instrumentation/libspu.la


instrumentation_libspu_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libspu_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libspu_la_SOURCES=$(spu_sources)
instrumentation_libspu_la_DEPENDENCIES=.libs/libspu_embed.a
instrumentation_libspu_la_LIBADD=$(spu_libadd)
endif

if is_instrumentation_debug_enabled
noinst_LTLIBRARIES += instrumentation-debug/libspu.la


instrumentation_debug_libspu_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libspu_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libspu_la_SOURCES=$(spu_sources)
instrumentation_debug_libspu_la_DEPENDENCIES=.libs/libspu_embed.a
instrumentation_debug_libspu_la_LIBADD=$(spu_libadd)
endif

if is_performance_enabled
noinst_LTLIBRARIES += performance/libspu.la


performance_libspu_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libspu_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libspu_la_SOURCES=$(spu_sources)
performance_libspu_la_DEPENDENCIES=.libs/libspu_embed.a
performance_libspu_la_LIBADD=$(spu_libadd)
endif


