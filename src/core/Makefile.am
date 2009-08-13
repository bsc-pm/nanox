
include $(top_srcdir)/src/common.am

lib_LTLIBRARIES=libnanox.la

if PERFORMANCE
  lib_LTLIBRARIES += libnanox-pfm.la
endif

# Common flags and files
nanos_cxxflags=$(common_includes) $(common_cxxflags)
nanos_ldflags=$(version_flags)
nanos_libadd=\
	-lpthread \
	-ldl \
	../arch/@OS@/libos.la \
	../support/libsupport.la \
	../sched/libsched.la \
	@NANOS_CONFIG_LIBS@\
	$(END)

nanos_sources=\
	basethread.hpp\
	basethread.cpp\
	processingelement.hpp \
	processingelement.cpp \
	schedule.hpp \
	schedule.cpp \
	system.hpp \
	system.cpp \
	wddeque.hpp \
	workgroup.hpp \
	workgroup.cpp \
	workdescriptor.hpp \
	workdescriptor.cpp \
	cutoff.hpp \
	$(END)

# Default
libnanox_la_CFLAGS=$(nanos_cflags)
libnanox_la_CXXFLAGS=$(nanos_cxxflags)
libnanox_la_LDFLAGS=$(nanos_ldflags) 
libnanox_la_LIBADD=$(nanos_libadd)
libnanox_la_SOURCES=$(nanos_sources)
libnanox_la_DEPENDENCIES=$(nanos_libadd)

# Performance
libnanox_pfm_la_CFLAGS=$(nanos_cflags) $(performance_cflags)
libnanox_pfm_la_CXXFLAGS=$(nanos_cxxflags) $(performance_cflags)
libnanox_pfm_la_LDFLAGS=$(nanos_ldflags) $(nanos_libadd)
libnanox_pfm_la_LIBADD=$(nanos_libadd)
libnanox_pfm_la_SOURCES=$(nanos_sources)
libnanox_pfm_la_DEPENDENCIES=$(nanos_libadd)
