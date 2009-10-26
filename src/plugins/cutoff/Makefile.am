
include $(top_srcdir)/src/common.am

lib_LTLIBRARIES = \
	libnanox-cutoff-tasknum.la \
	libnanox-cutoff-idle.la \
	libnanox-cutoff-level.la \
	libnanox-cutoff-ready.la \
	$(END)

noinst_LTLIBRARIES = libcutoff.la

libcutoff_la_CXXFLAGS= $(common_includes) $(common_cxxflags)
libcutoff_la_SOURCES = \
	dummy_cutoff.cpp \
	$(END)

libnanox_cutoff_tasknum_la_CXXFLAGS = $(common_includes) $(common_cxxflags)
libnanox_cutoff_tasknum_la_LDFLAGS = $(ld_plugin_flags)
libnanox_cutoff_tasknum_la_SOURCES = \
        tasknum_cutoff.cpp \
        $(END)

libnanox_cutoff_idle_la_CXXFLAGS = $(common_includes) $(common_cxxflags)
libnanox_cutoff_idle_la_LDFLAGS = $(ld_plugin_flags)
libnanox_cutoff_idle_la_SOURCES = \
        idle_cutoff.cpp \
        $(END)

libnanox_cutoff_level_la_CXXFLAGS = $(common_includes) $(common_cxxflags)
libnanox_cutoff_level_la_LDFLAGS = $(ld_plugin_flags)
libnanox_cutoff_level_la_SOURCES = \
        level_cutoff.cpp \
        $(END)

libnanox_cutoff_ready_la_CXXFLAGS = $(common_includes) $(common_cxxflags)
libnanox_cutoff_ready_la_LDFLAGS = $(ld_plugin_flags)
libnanox_cutoff_ready_la_SOURCES = \
        ready_cutoff.cpp \
        $(END)