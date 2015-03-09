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

EXTRA_DIST= \
	x86/stack.cpp \
	x86/stack_asm.s \
	x86-64/stack.cpp \
	x86-64/stack_asm.s \
	ia64/stack.cpp \
	ia64/stack_asm.s \
	ppc32/stack.cpp \
	ppc32/stack_asm.s \
	ppc64/stack.cpp \
	ppc64/stack_asm.s \
	ppc64_v2/stack.cpp \
	ppc64_v2/stack_asm.s \
	armv71/stack.cpp \
	armv71/stack_asm.s \
	tile64/stack.cpp \
	tile64/stack_asm.s \
	smp_ult.hpp \
	$(END)

devinclude_HEADERS = \
	smpdevice.hpp \
	smpdd.hpp \
	smpprocessor.hpp \
	smpprocessor_fwd.hpp \
	smpthread.hpp \
	smpthread_fwd.hpp \
	$(END) 

common_libadd=\
        $(END)

ult_sources= \
   @SMP_ARCH@/stack.cpp \
   @SMP_ARCH@/stack_asm.s \
   $(END)


ult_flags=-DSMP_SUPPORTS_ULT

smp_numa_flags=-DSMP_NUMA

smp_sources = \
	smpdevice.hpp \
	smpdd.hpp \
	smpdd.cpp \
	smpprocessor.hpp \
	smpprocessor_fwd.hpp \
	smpprocessor.cpp \
	smpthread.hpp \
	smpthread_fwd.hpp \
	smpthread.cpp \
	$(END)

pe_smp_sources = \
	smpplugin.cpp \
	$(END)


if SMP_SUPPORTS_ULT
smp_ult_sources=$(ult_sources)
smp_ult_flags=$(ult_flags)
endif

noinst_LTLIBRARIES = 

if DEBUG_VERSION
noinst_LTLIBRARIES += debug/libsmp.la
debug_LTLIBRARIES = debug/libnanox-pe-smp.la
debug_LTLIBRARIES += debug/libnanox-pe-smp-numa.la

nanos_debug_cxxflags=$(common_includes) $(common_debug_cxxflags) $(smp_ult_flags)

debug_libsmp_la_CXXFLAGS=$(nanos_debug_cxxflags)
debug_libsmp_la_CCASFLAGS=$(nanos_debug_cxxflags)
debug_libsmp_la_SOURCES=$(smp_sources) $(smp_ult_sources)

debug_libnanox_pe_smp_la_CXXFLAGS=$(nanos_debug_cxxflags)
debug_libnanox_pe_smp_la_LDFLAGS=$(ld_plugin_flags)
debug_libnanox_pe_smp_la_LIBADD=$(common_libadd)
debug_libnanox_pe_smp_la_SOURCES=$(pe_smp_sources)

debug_libnanox_pe_smp_numa_la_CXXFLAGS=$(nanos_debug_cxxflags) $(smp_numa_flags)
debug_libnanox_pe_smp_numa_la_LDFLAGS=$(ld_plugin_flags)
debug_libnanox_pe_smp_numa_la_LIBADD=$(common_libadd)
debug_libnanox_pe_smp_numa_la_SOURCES=$(pe_smp_sources)
endif

if INSTRUMENTATION_VERSION
noinst_LTLIBRARIES += instrumentation/libsmp.la
instrumentation_LTLIBRARIES = instrumentation/libnanox-pe-smp.la
instrumentation_LTLIBRARIES += instrumentation/libnanox-pe-smp-numa.la

nanos_instr_cxxflags=$(common_includes) $(common_instr_cxxflags) $(smp_ult_flags)

instrumentation_libsmp_la_CXXFLAGS=$(nanos_instr_cxxflags)
instrumentation_libsmp_la_CCASFLAGS=$(nanos_instr_cxxflags)
instrumentation_libsmp_la_SOURCES=$(smp_sources) $(smp_ult_sources)

instrumentation_libnanox_pe_smp_la_CXXFLAGS=$(nanos_instr_cxxflags)
instrumentation_libnanox_pe_smp_la_LDFLAGS=$(ld_plugin_flags)
instrumentation_libnanox_pe_smp_la_LIBADD=$(common_libadd)
instrumentation_libnanox_pe_smp_la_SOURCES=$(pe_smp_sources)

instrumentation_libnanox_pe_smp_numa_la_CXXFLAGS=$(nanos_instr_cxxflags) $(smp_numa_flags)
instrumentation_libnanox_pe_smp_numa_la_LDFLAGS=$(ld_plugin_flags)
instrumentation_libnanox_pe_smp_numa_la_LIBADD=$(common_libadd)
instrumentation_libnanox_pe_smp_numa_la_SOURCES=$(pe_smp_sources)
endif

if INSTRUMENTATION_DEBUG_VERSION
noinst_LTLIBRARIES += instrumentation-debug/libsmp.la
instrumentation_debug_LTLIBRARIES = instrumentation-debug/libnanox-pe-smp.la
instrumentation_debug_LTLIBRARIES += instrumentation-debug/libnanox-pe-smp-numa.la

nanos_instrdebug_cxxflags=$(common_includes) $(common_instrdebug_cxxflags) $(smp_ult_flags)

instrumentation_debug_libsmp_la_CXXFLAGS=$(nanos_instrdebug_cxxflags)
instrumentation_debug_libsmp_la_CCASFLAGS=$(nanos_instrdebug_cxxflags)
instrumentation_debug_libsmp_la_SOURCES=$(smp_sources) $(smp_ult_sources)

instrumentation_debug_libnanox_pe_smp_la_CXXFLAGS=$(nanos_instrdebug_cxxflags)
instrumentation_debug_libnanox_pe_smp_la_LDFLAGS=$(ld_plugin_flags)
instrumentation_debug_libnanox_pe_smp_la_LIBADD=$(common_libadd)
instrumentation_debug_libnanox_pe_smp_la_SOURCES=$(pe_smp_sources)

instrumentation_debug_libnanox_pe_smp_numa_la_CXXFLAGS=$(nanos_instrdebug_cxxflags) $(smp_numa_flags)
instrumentation_debug_libnanox_pe_smp_numa_la_LDFLAGS=$(ld_plugin_flags)
instrumentation_debug_libnanox_pe_smp_numa_la_LIBADD=$(common_libadd)
instrumentation_debug_libnanox_pe_smp_numa_la_SOURCES=$(pe_smp_sources)
endif

if PERFORMANCE_VERSION
noinst_LTLIBRARIES += performance/libsmp.la
performance_LTLIBRARIES = performance/libnanox-pe-smp.la
performance_LTLIBRARIES += performance/libnanox-pe-smp-numa.la

nanos_perfo_cxxflags=$(common_includes) $(common_perfo_cxxflags) $(smp_ult_flags)

performance_libsmp_la_CXXFLAGS=$(nanos_perfo_cxxflags)
performance_libsmp_la_CCASFLAGS=$(nanos_perfo_cxxflags)
performance_libsmp_la_SOURCES=$(smp_sources) $(smp_ult_sources)

performance_libnanox_pe_smp_la_CXXFLAGS=$(nanos_perfo_cxxflags)
performance_libnanox_pe_smp_la_LDFLAGS=$(ld_plugin_flags)
performance_libnanox_pe_smp_la_LIBADD=$(common_libadd)
performance_libnanox_pe_smp_la_SOURCES=$(pe_smp_sources)

performance_libnanox_pe_smp_numa_la_CXXFLAGS=$(nanos_perfo_cxxflags) $(smp_numa_flags)
performance_libnanox_pe_smp_numa_la_LDFLAGS=$(ld_plugin_flags)
performance_libnanox_pe_smp_numa_la_LIBADD=$(common_libadd)
performance_libnanox_pe_smp_numa_la_SOURCES=$(pe_smp_sources)
endif

