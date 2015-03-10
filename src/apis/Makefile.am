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

include ../common.am

common_includes += -I $(top_builddir)/src/apis/c

#####################################################################################
#### C ##############################################################################
#####################################################################################
pkginclude_HEADERS=\
	c/nanos.h \
   c/nanos_version.h \
   c/nanos_reduction.h \
   c/nanos_atomic.h \
	$(END)

BUILT_SOURCES=\
   c/nanos_version.h

EXTRA_DIST=\
   c/nanos_version.def

CLEANFILES = $(top_builddir)/src/apis/c/nanos_version.h 

c/nanos_version.h : $(srcdir)/c/nanos_version.def
	$(AM_V_GEN) (\
       echo "#ifndef  _NANOS_VERSION_H_" > $@; \
       echo "#define  _NANOS_VERSION_H_" >> $@; \
       cat $(srcdir)/c/nanos_version.def | $(SED) -e 's,^\s*\(\S\+\)\s*=\s*\(\S\+\)\s*$$,#define NANOS_API_\U\1 \2,' >> $@; \
       echo "#ifdef _MERCURIUM" >> $@; \
       echo "#ifndef _NANOS_INTERNAL" >> $@; \
       cat $(srcdir)/c/nanos_version.def | $(SED) -e 's,^\s*\(\S\+\)\s*=\s*\(\S\+\)\s*$$,#pragma nanos interface family(\1) version(\2),' >> $@; \
       echo "#endif // _NANOS_INTERNAL" >> $@; \
       echo "#endif // _MERCURIUM" >> $@; \
       echo "#endif // _NANOS_VERSION_H_" >> $@; \
       )


nanos_c_sources = \
	c/nanos.h\
	c/nanos_pe.cpp\
	c/nanos_finders.cpp\
	c/nanos_wd.cpp\
	c/nanos_worksharing.cpp\
	c/nanos_sync.cpp\
	c/nanos_team.cpp\
	c/nanos_err.cpp\
	c/nanos_sys.cpp\
	c/nanos_instrument.cpp\
	c/nanos_memory.cpp\
	c/nanos_atomic.h\
	c/nanos_atomic.cpp\
	c/nanos_reduction.h\
	c/nanos_reduction.cpp\
	c/nanos_sched.cpp\
	c/nanos_dependence.cpp\
	$(END) 


if DEBUG_VERSION
debug_LTLIBRARIES=debug/libnanox-c.la

nanos_debug_cxxflags=$(common_includes) $(common_debug_cxxflags)
nanos_debug_fcflags=$(common_includes) $(common_debug_fcflags)

debug_libnanox_c_la_FCFLAGS=$(nanos_debug_fcflags)
debug_libnanox_c_la_CXXFLAGS=$(nanos_debug_cxxflags)
debug_libnanox_c_la_LDFLAGS=$(version_flags) $(ignore_muldefs)
debug_libnanox_c_la_LIBADD=$(top_builddir)/src/core/debug/libnanox.la
debug_libnanox_c_la_SOURCES=$(nanos_c_sources)
debug_libnanox_c_la_DEPENDENCIES=$(top_builddir)/src/core/debug/libnanox.la
endif


if INSTRUMENTATION_VERSION
instrumentation_LTLIBRARIES=instrumentation/libnanox-c.la

nanos_instr_cxxflags=$(common_includes) $(common_instr_cxxflags)
nanos_instr_fcflags=$(common_includes) $(common_instr_fcflags)

instrumentation_libnanox_c_la_FCFLAGS=$(nanos_instr_fcflags)
instrumentation_libnanox_c_la_CXXFLAGS=$(nanos_instr_cxxflags)
instrumentation_libnanox_c_la_LDFLAGS=$(version_flags) $(ignore_muldefs)
instrumentation_libnanox_c_la_LIBADD=$(top_builddir)/src/core/instrumentation/libnanox.la
instrumentation_libnanox_c_la_SOURCES=$(nanos_c_sources)
instrumentation_libnanox_c_la_DEPENDENCIES=$(top_builddir)/src/core/instrumentation/libnanox.la
endif

if INSTRUMENTATION_DEBUG_VERSION
instrumentation_debug_LTLIBRARIES=instrumentation-debug/libnanox-c.la

nanos_instrdebug_cxxflags=$(common_includes) $(common_instrdebug_cxxflags)
nanos_instrdebug_fcflags=$(common_includes) $(common_instrdebug_fcflags)

instrumentation_debug_libnanox_c_la_FCFLAGS=$(nanos_instrdebug_fcflags)
instrumentation_debug_libnanox_c_la_CXXFLAGS=$(nanos_instrdebug_cxxflags)
instrumentation_debug_libnanox_c_la_LDFLAGS=$(version_flags) $(ignore_muldefs)
instrumentation_debug_libnanox_c_la_LIBADD=$(top_builddir)/src/core/instrumentation-debug/libnanox.la
instrumentation_debug_libnanox_c_la_SOURCES=$(nanos_c_sources)
instrumentation_debug_libnanox_c_la_DEPENDENCIES=$(top_builddir)/src/core/instrumentation-debug/libnanox.la
endif

if PERFORMANCE_VERSION
performance_LTLIBRARIES=performance/libnanox-c.la

nanos_perfo_cxxflags=$(common_includes) $(common_perfo_cxxflags)
nanos_perfo_fcflags=$(common_includes) $(common_perfo_fcflags)

performance_libnanox_c_la_FCFLAGS=$(nanos_perfo_fcflags)
performance_libnanox_c_la_CXXFLAGS=$(nanos_perfo_cxxflags)
performance_libnanox_c_la_LDFLAGS=$(version_flags) $(ignore_muldefs)
performance_libnanox_c_la_LIBADD=$(top_builddir)/src/core/performance/libnanox.la
performance_libnanox_c_la_SOURCES=$(nanos_c_sources)
performance_libnanox_c_la_DEPENDENCIES=$(top_builddir)/src/core/performance/libnanox.la
endif
#####################################################################################
#####################################################################################
