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

nanox_sources = \
   nanox.cpp \
   $(END)

bin_PROGRAMS =
if DEBUG_VERSION

bin_PROGRAMS += nanox-dbg

nanox_dbg_CPPFLAGS = $(common_includes) $(common_debug_cxxflags) $(bin_cxxflags) -DPLUGIN_DIR=\"$(debugdir)\"

nanox_dbg_SOURCES=  $(nanox_sources)

nanox_dbg_LDADD=\
	-ldl \
	$(top_builddir)/src/core/debug/libnanox.la \
	@CUDA_RPATH@ \
	$(END)

endif

if INSTRUMENTATION_VERSION

bin_PROGRAMS += nanox-ins

nanox_ins_CPPFLAGS = $(common_includes) $(common_instr_cxxflags) $(bin_cxxflags) -DPLUGIN_DIR=\"$(instrumentationdir)\"

nanox_ins_SOURCES=  $(nanox_sources)

nanox_ins_LDADD=\
	-ldl \
	$(top_builddir)/src/core/instrumentation/libnanox.la \
	@CUDA_RPATH@ \
	$(END)

endif

if INSTRUMENTATION_DEBUG_VERSION

bin_PROGRAMS += nanox-ins-dbg

nanox_ins_dbg_CPPFLAGS = $(common_includes) $(common_instrdebug_cxxflags) $(bin_cxxflags) -DPLUGIN_DIR=\"$(instrdebugdir)\"

nanox_ins_dbg_SOURCES=  $(nanox_sources)

nanox_ins_dbg_LDADD=\
	-ldl \
	$(top_builddir)/src/core/instrumentation-debug/libnanox.la \
	@CUDA_RPATH@ \
	$(END)

endif

if PERFORMANCE_VERSION

bin_PROGRAMS += nanox-pfm

nanox_pfm_CPPFLAGS = $(common_includes) $(common_perfo_cxxflags) $(bin_cxxflags) -DPLUGIN_DIR=\"$(performancedir)\"

nanox_pfm_SOURCES=  $(nanox_sources)

nanox_pfm_LDADD=\
	-ldl \
	$(top_builddir)/src/core/performance/libnanox.la \
	@CUDA_RPATH@ \
	$(END)

endif
