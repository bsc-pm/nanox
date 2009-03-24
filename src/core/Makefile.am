
include $(top_srcdir)/src/common.am

lib_LTLIBRARIES=libnanox.la

if PERFORMANCE
  lib_LTLIBRARIES += libnanox-pfm.la
endif

# Common flags and files
nanos_cxxflags=$(common_includes) $(common_cxxflags)
nanos_ldflags=-version-info 1:0:0 
nanos_libadd=\
	../arch/@OS@/libos.la \
	../support/libsupport.la \
	@NANOS_CONFIG_LIBS@\
	$(END)

nanos_sources=\
	coresetup.hpp \
	coresetup.cpp \
	processingelement.hpp \
	processingelement.cpp \
	system.hpp \
	system.cpp \
	workgroup.hpp \
	workgroup.cpp \
	workdescriptor.hpp \
	workdescriptor.cpp \
	$(END)

# Default
libnanox_la_CFLAGS=$(nanos_cflags)
libnanox_la_CXXFLAGS=$(nanos_cxxflags)
libnanox_la_LDFLAGS=$(nanos_ldflags) 
libnanox_la_LIBADD=$(nanos_libadd)
libnanox_la_SOURCES=$(nanos_sources)

# Performance
libnanox_pfm_la_CFLAGS=$(nanos_cflags) $(performance_cflags)
libnanox_pfm_la_CXXFLAGS=$(nanos_cxxflags) $(performance_cflags)
libnanox_pfm_la_LDFLAGS=$(nanos_ldflags) $(nanos_libadd)
libnanox_pfm_la_LIBADD=$(nanos_libadd)
libnanox_pfm_la_SOURCES=$(nanos_sources)
