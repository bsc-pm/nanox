
include $(top_srcdir)/src/common.am

noinst_LTLIBRARIES = libsmp.la
lib_LTLIBRARIES = libnanox-pe-smp.la

libsmp_la_CXXFLAGS=$(common_includes) $(common_cxxflags) 
libsmp_la_SOURCES = \
		smpdd.hpp \
		smpdd.cpp \
		@SMP_ARCH@/stack.cpp \
		@SMP_ARCH@/stack.s \
		$(END)


libnanox_pe_smp_la_CXXFLAGS=$(common_includes) $(common_cxxflags) 
libnanox_pe_smp_la_LDFLAGS = $(ld_plugin_flags)
libnanox_pe_smp_la_SOURCES = \
		smpplugin.cpp \
		smpprocessor.hpp \
		smpprocessor.cpp \
		smpthread.hpp \
		smpthread.cpp \
		$(END)

