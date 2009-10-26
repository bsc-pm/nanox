
include $(top_srcdir)/src/common.am

lib_LTLIBRARIES = \
	libnanox-barrier-centralized.la \
	$(END)

noinst_LTLIBRARIES = libbarrier.la

libbarrier_la_CXXFLAGS= $(common_includes) $(common_cxxflags)
libbarrier_la_SOURCES = \
	posix_barrier.cpp \
	dissem_barrier.cpp \
	$(END)


libnanox_barrier_centralized_la_CXXFLAGS = $(common_includes) $(common_cxxflags)
libnanox_barrier_centralized_la_LDFLAGS = $(ld_plugin_flags)
libnanox_barrier_centralized_la_SOURCES = \
	centralized_barrier.cpp \
	$(END)
