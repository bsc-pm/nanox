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

nanox_sources= \
   nanox.cpp \
   $(END)

bin_PROGRAMS=
if is_debug_enabled
bin_PROGRAMS += nanox-dbg

nanox_dbg_CPPFLAGS= $(common_debug_CPPFLAGS) -DPLUGIN_DIR=\"$(debugdir)\"
nanox_dbg_CXXFLAGS=$(common_debug_CXXFLAGS)
nanox_dbg_SOURCES= $(nanox_sources)
nanox_dbg_LDFLAGS=$(AM_LDFLAGS)
nanox_dbg_LDADD=\
-ldl \
$(top_builddir)/src/core/debug/libnanox.la \
$(END)

endif

if is_instrumentation_enabled
bin_PROGRAMS += nanox-ins

nanox_ins_CPPFLAGS= $(common_instrumentation_CPPFLAGS) -DPLUGIN_DIR=\"$(instrumentationdir)\"
nanox_ins_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
nanox_ins_SOURCES= $(nanox_sources)
nanox_ins_LDFLAGS=$(AM_LDFLAGS)
nanox_ins_LDADD=\
-ldl \
$(top_builddir)/src/core/instrumentation/libnanox.la \
$(END)

endif

if is_instrumentation_debug_enabled
bin_PROGRAMS += nanox-ins-dbg

nanox_ins_dbg_CPPFLAGS= $(common_instrumentation_debug_CPPFLAGS) -DPLUGIN_DIR=\"$(instrumentation_debugdir)\"
nanox_ins_dbg_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
nanox_ins_dbg_SOURCES= $(nanox_sources)
nanox_ins_dbg_LDFLAGS=$(AM_LDFLAGS)
nanox_ins_dbg_LDADD=\
	-ldl \
	$(top_builddir)/src/core/instrumentation-debug/libnanox.la \
	$(END)

endif

if is_performance_enabled
bin_PROGRAMS += nanox-pfm

nanox_pfm_CPPFLAGS= $(common_performance_CPPFLAGS) -DPLUGIN_DIR=\"$(performancedir)\"
nanox_pfm_CXXFLAGS=$(common_performance_CXXFLAGS)
nanox_pfm_SOURCES= $(nanox_sources)
nanox_pfm_LDFLAGS=$(AM_LDFLAGS)
nanox_pfm_LDADD=\
	-ldl \
	$(top_builddir)/src/core/performance/libnanox.la \
	$(END)

# For simplicity, we will only build nanox-bindings in PERFORMANCE version
bin_PROGRAMS += nanox-bindings
nanox_bindings_CPPFLAGS = $(common_performance_CPPFLAGS) $(common_includes) -DPLUGIN_DIR=\"$(performancedir)\"
nanox_bindings_CXXFLAGS=$(common_performance_CXXFLAGS)
nanox_bindings_SOURCES = nanox_bindings.cpp
nanox_bindings_LDFLAGS=$(AM_LDFLAGS)
nanox_bindings_LDADD = \
	$(top_builddir)/src/core/performance/libnanox.la \
	$(top_builddir)/src/pms/performance/libnanox-ompss.la \
	$(top_builddir)/src/apis/performance/libnanox-c.la \
	$(END)

endif
