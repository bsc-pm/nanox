
include $(top_srcdir)/src/common.am

noinst_LTLIBRARIES = libsmp.la

libsmp_la_CXXFLAGS=$(common_includes) $(common_cxxflags) 
libsmp_la_SOURCES = \
		smpprocessor.hpp \
		smpprocessor.cpp \
		smpthread.hpp \
		smpthread.cpp \
		smpwd.hpp \
		smpwd.cpp \
		@SMP_ARCH@/stack.cpp \
		@SMP_ARCH@/stack.s \
		$(END)

