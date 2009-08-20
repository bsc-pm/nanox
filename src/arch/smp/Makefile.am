
include $(top_srcdir)/src/common.am

noinst_LTLIBRARIES = libsmp.la

libsmp_la_CXXFLAGS=$(common_includes) $(common_cxxflags) 
libsmp_la_SOURCES = \
		smpdd.hpp \
		smpdd.cpp \
		smpprocessor.hpp \
		smpprocessor.cpp \
		smpthread.hpp \
		smpthread.cpp \
		@SMP_ARCH@/stack.cpp \
		@SMP_ARCH@/stack.s \
		$(END)

