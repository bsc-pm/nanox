include $(top_srcdir)/src/common.am

noinst_PROGRAMS = test1

test1_SOURCES = test1.cpp
test1_CXXFLAGS = -g $(common_includes) $(common_cxxflags)
test1_LDADD = $(top_builddir)/src/core/libnanox.la -lpthread


