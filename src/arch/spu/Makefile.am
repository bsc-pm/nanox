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

include ../../common.am

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

if DEBUG_VERSION
noinst_LTLIBRARIES += debug/libspu.la

nanos_debug_cxxflags=$(common_includes) $(common_debug_cxxflags) $(spu_ccflags)

debug_libspu_la_CXXFLAGS=$(nanos_debug_cxxflags)
debug_libspu_la_SOURCES=$(spu_sources)
debug_libspu_la_DEPENDENCIES=.libs/libspu_embed.a
debug_libspu_la_LIBADD=$(spu_libadd)
endif

if INSTRUMENTATION_VERSION
noinst_LTLIBRARIES += instrumentation/libspu.la

nanos_instr_cxxflags=$(common_includes) $(common_instr_cxxflags) $(spu_ccflags)

instrumentation_libspu_la_CXXFLAGS=$(nanos_instr_cxxflags)
instrumentation_libspu_la_SOURCES=$(spu_sources)
instrumentation_libspu_la_DEPENDENCIES=.libs/libspu_embed.a
instrumentation_libspu_la_LIBADD=$(spu_libadd)
endif

if INSTRUMENTATION_DEBUG_VERSION
noinst_LTLIBRARIES += instrumentation-debug/libspu.la

nanos_instrdebug_cxxflags=$(common_includes) $(common_instrdebug_cxxflags) $(spu_ccflags)

instrumentation_debug_libspu_la_CXXFLAGS=$(nanos_instrdebug_cxxflags)
instrumentation_debug_libspu_la_SOURCES=$(spu_sources)
instrumentation_debug_libspu_la_DEPENDENCIES=.libs/libspu_embed.a
instrumentation_debug_libspu_la_LIBADD=$(spu_libadd)
endif

if PERFORMANCE_VERSION
noinst_LTLIBRARIES += performance/libspu.la

nanos_perfo_cxxflags=$(common_includes) $(common_perfo_cxxflags) $(spu_ccflags)

performance_libspu_la_CXXFLAGS=$(nanos_perfo_cxxflags)
performance_libspu_la_SOURCES=$(spu_sources)
performance_libspu_la_DEPENDENCIES=.libs/libspu_embed.a
performance_libspu_la_LIBADD=$(spu_libadd)
endif


