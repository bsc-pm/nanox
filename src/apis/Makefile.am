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

#####################################################################################
#### C ##############################################################################
#####################################################################################
pkginclude_HEADERS=\
c/nanos.h \
c/nanos_version.h \
c/nanos_atomic.h \
$(END)

BUILT_SOURCES=\
c/nanos_version.h

EXTRA_DIST=\
c/nanos_version.def

CLEANFILES = $(top_builddir)/src/apis/c/nanos_version.h 

c/nanos_version.h : $(srcdir)/c/nanos_version.def
	$(AM_V_GEN) (\
       echo "#ifndef  _NANOS_enabled_H_" > $@; \
       echo "#define  _NANOS_enabled_H_" >> $@; \
       cat $(srcdir)/c/nanos_version.def | $(SED) -e 's,^\s*\(\S\+\)\s*=\s*\(\S\+\)\s*$$,#define NANOS_API_\U\1 \2,' >> $@; \
       echo "#ifdef _MERCURIUM" >> $@; \
       echo "#ifndef _NANOS_INTERNAL" >> $@; \
       cat $(srcdir)/c/nanos_version.def | $(SED) -e 's,^\s*\(\S\+\)\s*=\s*\(\S\+\)\s*$$,#pragma nanos interface family(\1) version(\2),' >> $@; \
       echo "#endif /* _NANOS_INTERNAL */" >> $@; \
       echo "#endif /* _MERCURIUM */" >> $@; \
       echo "#endif /* _NANOS_enabled_H_ */" >> $@; \
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
	c/nanos_sched.cpp\
	c/nanos_dependence.cpp\
	c/iomp_symbols.cpp\
	$(END) 


if is_debug_enabled
debug_LTLIBRARIES=debug/libnanox-c.la

debug_libnanox_c_la_CPPFLAGS=$(common_debug_CPPFLAGS) $(api_includes)
debug_libnanox_c_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_c_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
debug_libnanox_c_la_LIBADD=$(top_builddir)/src/core/debug/libnanox.la
debug_libnanox_c_la_SOURCES=$(nanos_c_sources)
debug_libnanox_c_la_DEPENDENCIES=$(top_builddir)/src/core/debug/libnanox.la
endif

if is_instrumentation_debug_enabled
instrumentation_debug_LTLIBRARIES=instrumentation-debug/libnanox-c.la

instrumentation_debug_libnanox_c_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) $(api_includes)
instrumentation_debug_libnanox_c_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_c_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
instrumentation_debug_libnanox_c_la_LIBADD=$(top_builddir)/src/core/instrumentation-debug/libnanox.la
instrumentation_debug_libnanox_c_la_SOURCES=$(nanos_c_sources)
instrumentation_debug_libnanox_c_la_DEPENDENCIES=$(top_builddir)/src/core/instrumentation-debug/libnanox.la
endif

if is_instrumentation_enabled
instrumentation_LTLIBRARIES=instrumentation/libnanox-c.la

instrumentation_libnanox_c_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) $(api_includes)
instrumentation_libnanox_c_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_c_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
instrumentation_libnanox_c_la_LIBADD=$(top_builddir)/src/core/instrumentation/libnanox.la
instrumentation_libnanox_c_la_SOURCES=$(nanos_c_sources)
instrumentation_libnanox_c_la_DEPENDENCIES=$(top_builddir)/src/core/instrumentation/libnanox.la
endif

if is_performance_enabled
performance_LTLIBRARIES=performance/libnanox-c.la

performance_libnanox_c_la_CPPFLAGS=$(common_performance_CPPFLAGS) $(api_includes)
performance_libnanox_c_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_c_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
performance_libnanox_c_la_LIBADD=$(top_builddir)/src/core/performance/libnanox.la
performance_libnanox_c_la_SOURCES=$(nanos_c_sources)
performance_libnanox_c_la_DEPENDENCIES=$(top_builddir)/src/core/performance/libnanox.la
endif

#####################################################################################
#####################################################################################
