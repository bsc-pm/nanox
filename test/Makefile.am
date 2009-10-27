include $(top_srcdir)/src/common.am

noinst_PROGRAMS = test1 test_single test_barrier fib fib-c fibnocutoff-c

test1_SOURCES = test1.cpp
test1_CXXFLAGS = -g $(common_includes) $(common_cxxflags)
test1_LDADD = $(top_builddir)/src/core/libnanox.la

test_single_SOURCES = test_single.cpp
test_single_CXXFLAGS = -g $(common_includes) $(common_cxxflags) -I$(top_builddir)/src/apis/c
test_single_LDADD = $(top_builddir)/src/core/libnanox.la $(top_builddir)/src/apis/c/libnanox-c.la

test_barrier_SOURCES = test_barrier.cpp
test_barrier_CXXFLAGS = -g $(common_includes) $(common_cxxflags)
test_barrier_LDADD = $(top_builddir)/src/core/libnanox.la

fib_SOURCES = fib.cpp
fib_CXXFLAGS = -g $(common_includes) $(common_cxxflags)
fib_LDADD = $(top_builddir)/src/core/libnanox.la 

fib_c_SOURCES = fib.c
fib_c_CFLAGS = -g $(common_includes) $(common_cxxflags) -I $(top_builddir)/src/apis/c -fno-strict-aliasing
fib_c_LDADD = $(top_builddir)/src/core/libnanox.la $(top_builddir)/src/apis/c/libnanox-c.la 

fibnocutoff_c_SOURCES = fibnocutoff.c
fibnocutoff_c_CFLAGS = -g $(common_includes) $(common_cxxflags) -I $(top_builddir)/src/apis/c -fno-strict-aliasing
fibnocutoff_c_LDADD = $(top_builddir)/src/core/libnanox.la $(top_builddir)/src/apis/c/libnanox-c.la 


