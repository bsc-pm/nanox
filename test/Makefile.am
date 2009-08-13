include $(top_srcdir)/src/common.am

noinst_PROGRAMS = test1 fib fib-c

test1_SOURCES = test1.cpp
test1_CXXFLAGS = -g $(common_includes) $(common_cxxflags)
test1_LDADD = $(top_builddir)/src/core/libnanox.la

fib_SOURCES = fib.cpp
fib_CXXFLAGS = -g $(common_includes) $(common_cxxflags)
fib_LDADD = $(top_builddir)/src/core/libnanox.la 

fib_c_SOURCES = fib.c
fib_c_CFLAGS = -g $(common_includes) $(common_cxxflags) -I $(top_builddir)/src/apis/c
fib_c_LDADD = $(top_builddir)/src/core/libnanox.la $(top_builddir)/src/apis/c/libnanox-c.la 


