include $(top_srcdir)/src/common.am

lib_LTLIBRARIES=libnanox-c.la

include_HEADERS=\
	nanos.h
	$(END)

sources=\
	nanos.h\
	nanos_wd.cpp\
	nanos_wg.cpp\
	nanos_team.cpp\
	nanos_err.cpp\
	nanos_sys.cpp\
	$(END)

libnanox_c_la_CXXFLAGS=$(common_includes) $(common_cxxflags) $(performance_cflags)
libnanox_c_la_LDFLAGS=$(version_flags)
libnanox_c_la_LIBADD=$(top_builddir)/src/core/libnanox.la
libnanox_c_la_SOURCES=$(sources)
libnanox_c_la_DEPENDENCIES=$(top_builddir)/src/core/libnanox.la

