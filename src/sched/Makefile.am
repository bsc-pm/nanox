
include $(top_srcdir)/src/common.am

lib_LTLIBRARIES = \
	libnanox-sched-cilk.la \
	libnanox-sched-wf.la \
	libnanox-sched-bf.la \
	libnanox-cutoff-tasknum.la \
	libnanox-cutoff-idle.la \
	libnanox-cutoff-level.la \
	libnanox-cutoff-ready.la \
	$(END)

noinst_LTLIBRARIES = libsched.la libcutoff.la libbarrier.la

libsched_la_CXXFLAGS= $(common_includes) $(common_cxxflags)
libsched_la_SOURCES = \
	$(END) 

libcutoff_la_CXXFLAGS= $(common_includes) $(common_cxxflags)
libcutoff_la_SOURCES = \
	dummy_cutoff.cpp \
	$(END)

libbarrier_la_CXXFLAGS= $(common_includes) $(common_cxxflags)
libbarrier_la_SOURCES = \
	centralized_barrier.cpp \
	posix_barrier.cpp \
	dissem_barrier.cpp \
	$(END) 

libnanox_sched_bf_la_CXXFLAGS = $(common_includes) $(common_cxxflags)
libnanox_sched_bf_la_LDFLAGS = $(ld_plugin_flags)
libnanox_sched_bf_la_SOURCES = \
	bf_sched.cpp \
	$(END)

libnanox_sched_wf_la_CXXFLAGS = $(common_includes) $(common_cxxflags)
libnanox_sched_wf_la_LDFLAGS = $(ld_plugin_flags)
libnanox_sched_wf_la_SOURCES = \
	wf_sched.cpp \
	$(END)

libnanox_sched_cilk_la_CXXFLAGS = $(common_includes) $(common_cxxflags)
libnanox_sched_cilk_la_LDFLAGS = $(ld_plugin_flags)
libnanox_sched_cilk_la_SOURCES = \
	cilk_sched.cpp \
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