
include $(top_srcdir)/src/common.am

lib_LTLIBRARIES = libnanox-sched-cilk.la libnanox-sched-bf.la

noinst_LTLIBRARIES = libsched.la

libsched_la_CXXFLAGS= $(common_includes) $(common_cxxflags)
libsched_la_SOURCES = \
	wf_sched.cpp \
	dummy_cutoff.cpp \
	tasknum_cutoff.cpp \
	$(END) 

libnanox_sched_bf_la_CXXFLAGS = $(common_includes) $(common_cxxflags)
libnanox_sched_bf_la_LDFLAGS = $(ld_plugin_flags)
libnanox_sched_bf_la_SOURCES = \
	bf_sched.cpp \
	$(END)

libnanox_sched_cilk_la_CXXFLAGS = $(common_includes) $(common_cxxflags)
libnanox_sched_cilk_la_LDFLAGS = $(ld_plugin_flags)
libnanox_sched_cilk_la_SOURCES = \
	cilk_sched.cpp \
	$(END)
