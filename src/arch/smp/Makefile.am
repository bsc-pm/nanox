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
	armv71_hf/stack.cpp \
	armv71_hf/stack_asm.s \
	aarch64/stack.cpp \
	aarch64/stack_asm.s \
	tile64/stack.cpp \
	tile64/stack_asm.s \
	riscv64/stack.cpp \
	riscv64/stack_asm.s \
	smp_ult.hpp \
	$(END)

devinclude_HEADERS = \
	smpdevice.hpp \
	smpdevice_decl.hpp \
	smpdd.hpp \
	smpprocessor.hpp \
	smpprocessor_fwd.hpp \
	smpthread.hpp \
	smpthread_fwd.hpp \
	smptransferqueue_decl.hpp \
	smptransferqueue.hpp \
	$(END)

common_libadd=\
        $(END)

smp_sources = \
	smpdevice.hpp \
	smpdevice_decl.hpp \
	smptransferqueue.hpp \
	smptransferqueue_decl.hpp \
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
	smpplugin_decl.hpp \
	smpplugin.cpp \
	$(END)

if SMP_SUPPORTS_ULT
  ult_sources= \
@SMP_ARCH@/stack.cpp \
@SMP_ARCH@/stack_asm.s \
$(END)
else
  ult_sources=
endif

noinst_LTLIBRARIES =

if is_debug_enabled
noinst_LTLIBRARIES += debug/libsmp.la
debug_LTLIBRARIES = debug/libnanox-pe-smp.la

debug_libsmp_la_CPPFLAGS=$(common_debug_CPPFLAGS)
debug_libsmp_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libsmp_la_CCASFLAGS=$(common_debug_CCASFLAGS)
debug_libsmp_la_SOURCES=$(smp_sources) $(ult_sources)

debug_libnanox_pe_smp_la_CPPFLAGS=$(common_debug_CPPFLAGS) @memkindinc@
debug_libnanox_pe_smp_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_pe_smp_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @memkindlib@
debug_libnanox_pe_smp_la_LIBADD=$(common_libadd) @memkindlibs@
debug_libnanox_pe_smp_la_SOURCES=$(pe_smp_sources)
endif

if is_instrumentation_enabled
noinst_LTLIBRARIES += instrumentation/libsmp.la
instrumentation_LTLIBRARIES = instrumentation/libnanox-pe-smp.la

instrumentation_libsmp_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS)
instrumentation_libsmp_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libsmp_la_CCASFLAGS=$(common_instrumentation_CCASFLAGS)
instrumentation_libsmp_la_SOURCES=$(smp_sources) $(ult_sources)

instrumentation_libnanox_pe_smp_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) @memkindinc@
instrumentation_libnanox_pe_smp_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_pe_smp_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @memkindlib@
instrumentation_libnanox_pe_smp_la_LIBADD=$(common_libadd) @memkindlibs@
instrumentation_libnanox_pe_smp_la_SOURCES=$(pe_smp_sources)
endif

if is_instrumentation_debug_enabled
noinst_LTLIBRARIES += instrumentation-debug/libsmp.la
instrumentation_debug_LTLIBRARIES = instrumentation-debug/libnanox-pe-smp.la

instrumentation_debug_libsmp_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS)
instrumentation_debug_libsmp_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libsmp_la_CCASFLAGS=$(common_instrumentation_debug_CCASFLAGS)
instrumentation_debug_libsmp_la_SOURCES=$(smp_sources) $(ult_sources)

instrumentation_debug_libnanox_pe_smp_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) @memkindinc@
instrumentation_debug_libnanox_pe_smp_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_pe_smp_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @memkindlib@
instrumentation_debug_libnanox_pe_smp_la_LIBADD=$(common_libadd) @memkindlibs@
instrumentation_debug_libnanox_pe_smp_la_SOURCES=$(pe_smp_sources)
endif

if is_performance_enabled
noinst_LTLIBRARIES += performance/libsmp.la
performance_LTLIBRARIES = performance/libnanox-pe-smp.la

performance_libsmp_la_CPPFLAGS=$(common_performance_CPPFLAGS)
performance_libsmp_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libsmp_la_CCASFLAGS=$(common_performance_CCASFLAGS)
performance_libsmp_la_SOURCES=$(smp_sources) $(ult_sources)

performance_libnanox_pe_smp_la_CPPFLAGS=$(common_performance_CPPFLAGS) @memkindinc@
performance_libnanox_pe_smp_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_pe_smp_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @memkindlib@
performance_libnanox_pe_smp_la_LIBADD=$(common_libadd) @memkindlibs@
performance_libnanox_pe_smp_la_SOURCES=$(pe_smp_sources)
endif
