#####################################################################################
#      Copyright 2009 Barcelona Supercomputing Center                               #
#                                                                                   #
#      This file is part of the NANOS++ library.                                    #
#                                                                                   #
#      NANOS++ is free software: you can redistribute it and/or modify              #
#      it under the terms of the GNU Lesser General Public License as published by  #
#      the Free Software Foundation, either version 3 of the License, or            #
#      (at your option) any later version.                                          #
#                                                                                   #
#      NANOS++ is distributed in the hope that it will be useful,                   #
#      but WITHOUT ANY WARRANTY; without even the implied warranty of               #
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                #
#      GNU Lesser General Public License for more details.                          #
#                                                                                   #
#      You should have received a copy of the GNU Lesser General Public License     #
#      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             #
#####################################################################################

include $(top_srcdir)/src/common.am

noinst_PROGRAMS = test1 test_single test_teambarrier test_barrier fib fib-c fibnocutoff-c

test1_SOURCES = test1.cpp
test1_CXXFLAGS = -g $(common_includes) $(common_cxxflags)
test1_LDADD = $(top_builddir)/src/core/libnanox.la

test_single_SOURCES = test_single.cpp
test_single_CXXFLAGS = -g $(common_includes) $(common_cxxflags) -I$(top_builddir)/src/apis/c
test_single_LDADD = $(top_builddir)/src/core/libnanox.la $(top_builddir)/src/apis/c/libnanox-c.la

test_teambarrier_SOURCES = test_teambarrier.cpp
test_teambarrier_CXXFLAGS = -g $(common_includes) $(common_cxxflags) -I$(top_builddir)/src/apis/c
test_teambarrier_LDADD = $(top_builddir)/src/core/libnanox.la $(top_builddir)/src/apis/c/libnanox-c.la

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


