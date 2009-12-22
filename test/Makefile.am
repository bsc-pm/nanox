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

noinst_PROGRAMS = \
	work_descriptor_create \
	slicer_for \
	api_slicer_for \
	test1 test_single test_slicer test_teambarrier test_dependencies-c test_dependency_types-c fib fib-c fibnocutoff-c

work_descriptor_create_SOURCES = work_descriptor_create.cpp
work_descriptor_create_CXXFLAGS = -g $(common_includes) $(common_cxxflags) -I$(top_srcdir)/src/apis/c
work_descriptor_create_LDADD = $(top_builddir)/src/core/libnanox.la

slicer_for_SOURCES = slicer_for.cpp
slicer_for_CXXFLAGS = -g $(common_includes) $(common_cxxflags) -I$(top_srcdir)/src/apis/c
slicer_for_LDADD = $(top_builddir)/src/core/libnanox.la

api_slicer_for_SOURCES = api_slicer_for.c
api_slicer_for_CFLAGS = -g $(common_includes) $(common_cflags) -I $(top_srcdir)/src/apis/c -fno-strict-aliasing
api_slicer_for_LDADD = $(top_builddir)/src/core/libnanox.la $(top_builddir)/src/apis/c/libnanox-c.la 

test1_SOURCES = test1.cpp
test1_CXXFLAGS = -g $(common_includes) $(common_cxxflags)
test1_LDADD = $(top_builddir)/src/core/libnanox.la

test_single_SOURCES = test_single.cpp
test_single_CXXFLAGS = -g $(common_includes) $(common_cxxflags) -I$(top_srcdir)/src/apis/c
test_single_LDADD = $(top_builddir)/src/core/libnanox.la $(top_builddir)/src/apis/c/libnanox-c.la

test_slicer_SOURCES = test_slicer.cpp
test_slicer_CXXFLAGS = -g $(common_includes) $(common_cxxflags)
test_slicer_LDADD = $(top_builddir)/src/core/libnanox.la

test_teambarrier_SOURCES = test_teambarrier.cpp
test_teambarrier_CXXFLAGS = -g $(common_includes) $(common_cxxflags) -I$(top_srcdir)/src/apis/c
test_teambarrier_LDADD = $(top_builddir)/src/core/libnanox.la $(top_builddir)/src/apis/c/libnanox-c.la

test_dependencies_c_SOURCES = test_dependencies.c
test_dependencies_c_CFLAGS = -g $(common_includes) $(common_cflags) -I $(top_srcdir)/src/apis/c -fno-strict-aliasing
test_dependencies_c_LDADD = $(top_builddir)/src/core/libnanox.la $(top_builddir)/src/apis/c/libnanox-c.la

test_dependency_types_c_SOURCES = test_dependency_types.c
test_dependency_types_c_CFLAGS = -g $(common_includes) $(common_cflags) -I $(top_srcdir)/src/apis/c -fno-strict-aliasing
test_dependency_types_c_LDADD = $(top_builddir)/src/core/libnanox.la $(top_builddir)/src/apis/c/libnanox-c.la

fib_SOURCES = fib.cpp
fib_CXXFLAGS = -g $(common_includes) $(common_cxxflags)
fib_LDADD = $(top_builddir)/src/core/libnanox.la 

fib_c_SOURCES = fib.c
fib_c_CFLAGS = -g $(common_includes) $(common_cflags) -I $(top_srcdir)/src/apis/c -fno-strict-aliasing
fib_c_LDADD = $(top_builddir)/src/core/libnanox.la $(top_builddir)/src/apis/c/libnanox-c.la 

fibnocutoff_c_SOURCES = fibnocutoff.c
fibnocutoff_c_CFLAGS = -g $(common_includes) $(common_cflags) -I $(top_srcdir)/src/apis/c -fno-strict-aliasing
fibnocutoff_c_LDADD = $(top_builddir)/src/core/libnanox.la $(top_builddir)/src/apis/c/libnanox-c.la 


