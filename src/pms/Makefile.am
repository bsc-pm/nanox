#####################################################################################
#      Copyright 2015 Barcelona Supercomputing Center                               #
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

pkginclude_HEADERS=
debug_LTLIBRARIES=
instrumentation_LTLIBRARIES=
instrumentation_debug_LTLIBRARIES=
performance_LTLIBRARIES=


##### OpenMP ######################################

omp_sources = \
   openmp/omp_init.cpp\
   openmp/omp_init.hpp\
   openmp/omp.h\
   openmp/omp_wd_data.hpp\
   openmp/omp_threadteam_data.hpp\
   openmp/omp_api.cpp\
   openmp/omp_time.cpp\
   openmp/omp_locks.cpp\
   openmp/nanos_omp.cpp\
   openmp/nanos_omp.h\
	$(END) 

omp_stub_sources = \
   openmp/omp.h\
   openmp/omp_stubs.cpp\
   openmp/omp_time.cpp\
   openmp/omp_locks.cpp\
	$(END) 

pkginclude_HEADERS+=\
   openmp/omp.h \
   openmp/nanos_omp.h \
	$(END)

if is_debug_enabled
debug_LTLIBRARIES+=\
 debug/libnanox-omp.la\
 debug/libnanox-omp-stubs.la
 
debug_LTLIBRARIES+=\
 debug/libnanox-ompss.la\
 debug/libnanox-ompss-stubs.la

debug_libnanox_omp_la_CPPFLAGS=$(common_debug_CPPFLAGS) $(api_includes) # @cudainc@
debug_libnanox_omp_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_omp_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
debug_libnanox_omp_la_LIBADD=$(top_builddir)/src/core/debug/libnanox.la
debug_libnanox_omp_la_SOURCES=$(omp_sources)
debug_libnanox_omp_la_DEPENDENCIES=$(top_builddir)/src/core/debug/libnanox.la

debug_libnanox_ompss_la_CPPFLAGS=$(common_debug_CPPFLAGS) $(api_includes) -DNANOX_SS_SUPPORT # @cudainc@
debug_libnanox_ompss_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_ompss_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
debug_libnanox_ompss_la_LIBADD=$(top_builddir)/src/core/debug/libnanox.la
debug_libnanox_ompss_la_SOURCES=$(omp_sources)
debug_libnanox_ompss_la_DEPENDENCIES=$(top_builddir)/src/core/debug/libnanox.la

debug_libnanox_omp_stubs_la_CPPFLAGS=$(common_debug_CPPFLAGS) $(api_includes)  # @cudainc@
debug_libnanox_omp_stubs_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_omp_stubs_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
debug_libnanox_omp_stubs_la_SOURCES=$(omp_stub_sources)

debug_libnanox_ompss_stubs_la_CPPFLAGS=$(common_debug_CPPFLAGS) $(api_includes) @cudainc@
debug_libnanox_ompss_stubs_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_ompss_stubs_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
debug_libnanox_ompss_stubs_la_SOURCES=$(omp_stub_sources)
endif

if is_instrumentation_debug_enabled
instrumentation_debug_LTLIBRARIES+=\
 instrumentation-debug/libnanox-omp.la\
 instrumentation-debug/libnanox-omp-stubs.la
 
instrumentation_debug_LTLIBRARIES+=\
 instrumentation-debug/libnanox-ompss.la\
 instrumentation-debug/libnanox-ompss-stubs.la

instrumentation_debug_libnanox_omp_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) $(api_includes) # @cudainc@
instrumentation_debug_libnanox_omp_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_omp_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
instrumentation_debug_libnanox_omp_la_LIBADD=$(top_builddir)/src/core/instrumentation-debug/libnanox.la
instrumentation_debug_libnanox_omp_la_SOURCES=$(omp_sources)
instrumentation_debug_libnanox_omp_la_DEPENDENCIES=$(top_builddir)/src/core/instrumentation-debug/libnanox.la

instrumentation_debug_libnanox_ompss_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) $(api_includes) -DNANOX_SS_SUPPORT # @cudainc@
instrumentation_debug_libnanox_ompss_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_ompss_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
instrumentation_debug_libnanox_ompss_la_LIBADD=$(top_builddir)/src/core/instrumentation-debug/libnanox.la
instrumentation_debug_libnanox_ompss_la_SOURCES=$(omp_sources)
instrumentation_debug_libnanox_ompss_la_DEPENDENCIES=$(top_builddir)/src/core/instrumentation-debug/libnanox.la

instrumentation_debug_libnanox_omp_stubs_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) $(api_includes)  # @cudainc@
instrumentation_debug_libnanox_omp_stubs_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_omp_stubs_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
instrumentation_debug_libnanox_omp_stubs_la_SOURCES=$(omp_stub_sources)

instrumentation_debug_libnanox_ompss_stubs_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) $(api_includes) @cudainc@
instrumentation_debug_libnanox_ompss_stubs_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_ompss_stubs_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
instrumentation_debug_libnanox_ompss_stubs_la_SOURCES=$(omp_stub_sources)
endif

if is_instrumentation_enabled
instrumentation_LTLIBRARIES+=\
 instrumentation/libnanox-omp.la\
 instrumentation/libnanox-omp-stubs.la
 
instrumentation_LTLIBRARIES+=\
 instrumentation/libnanox-ompss.la\
 instrumentation/libnanox-ompss-stubs.la

instrumentation_libnanox_omp_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) $(api_includes) # @cudainc@
instrumentation_libnanox_omp_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_omp_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
instrumentation_libnanox_omp_la_LIBADD=$(top_builddir)/src/core/instrumentation/libnanox.la
instrumentation_libnanox_omp_la_SOURCES=$(omp_sources)
instrumentation_libnanox_omp_la_DEPENDENCIES=$(top_builddir)/src/core/instrumentation/libnanox.la

instrumentation_libnanox_ompss_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) $(api_includes) -DNANOX_SS_SUPPORT # @cudainc@
instrumentation_libnanox_ompss_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_ompss_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
instrumentation_libnanox_ompss_la_LIBADD=$(top_builddir)/src/core/instrumentation/libnanox.la
instrumentation_libnanox_ompss_la_SOURCES=$(omp_sources)
instrumentation_libnanox_ompss_la_DEPENDENCIES=$(top_builddir)/src/core/instrumentation/libnanox.la

instrumentation_libnanox_omp_stubs_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) $(api_includes)  # @cudainc@
instrumentation_libnanox_omp_stubs_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_omp_stubs_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
instrumentation_libnanox_omp_stubs_la_SOURCES=$(omp_stub_sources)

instrumentation_libnanox_ompss_stubs_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) $(api_includes) @cudainc@
instrumentation_libnanox_ompss_stubs_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_ompss_stubs_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
instrumentation_libnanox_ompss_stubs_la_SOURCES=$(omp_stub_sources)
endif

if is_performance_enabled
performance_LTLIBRARIES+=\
 performance/libnanox-omp.la\
 performance/libnanox-omp-stubs.la
 
performance_LTLIBRARIES+=\
 performance/libnanox-ompss.la\
 performance/libnanox-ompss-stubs.la

performance_libnanox_omp_la_CPPFLAGS=$(common_performance_CPPFLAGS) $(api_includes) # @cudainc@
performance_libnanox_omp_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_omp_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
performance_libnanox_omp_la_LIBADD=$(top_builddir)/src/core/performance/libnanox.la
performance_libnanox_omp_la_SOURCES=$(omp_sources)
performance_libnanox_omp_la_DEPENDENCIES=$(top_builddir)/src/core/performance/libnanox.la

performance_libnanox_ompss_la_CPPFLAGS=$(common_performance_CPPFLAGS) $(api_includes) -DNANOX_SS_SUPPORT # @cudainc@
performance_libnanox_ompss_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_ompss_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
performance_libnanox_ompss_la_LIBADD=$(top_builddir)/src/core/performance/libnanox.la
performance_libnanox_ompss_la_SOURCES=$(omp_sources)
performance_libnanox_ompss_la_DEPENDENCIES=$(top_builddir)/src/core/performance/libnanox.la

performance_libnanox_omp_stubs_la_CPPFLAGS=$(common_performance_CPPFLAGS) $(api_includes)  # @cudainc@
performance_libnanox_omp_stubs_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_omp_stubs_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
performance_libnanox_omp_stubs_la_SOURCES=$(omp_stub_sources)

performance_libnanox_ompss_stubs_la_CPPFLAGS=$(common_performance_CPPFLAGS) $(api_includes) @cudainc@
performance_libnanox_ompss_stubs_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_ompss_stubs_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
performance_libnanox_ompss_stubs_la_SOURCES=$(omp_stub_sources)
endif


###################################################

#### Chapel #######################################

if CHAPEL_SUPPORT

chpl_sources = \
   chapel/chpl_nanos.h\
   chapel/chpl_main.cpp\
   chapel/chpl_sync.cpp\
   $(END) 

chpl_flags=-I@CHAPEL_HOME@/runtime/include

pkginclude_HEADERS+=\
   chapel/chpl_nanos.h\
	$(END)

lib_LIBRARIES=libnanox-chpl-init.a

libnanox_chpl_init_a_SOURCES= chapel/chpl_init.c
libnanox_chpl_init_a_CPPFLAGS=$(AM_CPPFLAGS) $(chpl_flags)
if is_debug_enabled
debug_LTLIBRARIES+=debug/libnanox-chpl.la

debug_libnanox_chpl_la_CPPFLAGS=$(common_debug_CPPFLAGS) $(chpl_flags)
debug_libnanox_chpl_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_chpl_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
debug_libnanox_chpl_la_LIBADD=debug/libnanox.la
debug_libnanox_chpl_la_SOURCES=$(chpl_sources)
debug_libnanox_chpl_la_DEPENDENCIES=debug/libnanox.la
endif

if is_instrumentation_enabled
instrumentation_LTLIBRARIES+=instrumentation/libnanox-chpl.la

instrumentation_libnanox_chpl_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) $(chpl_flags)
instrumentation_libnanox_chpl_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_chpl_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
instrumentation_libnanox_chpl_la_LIBADD=instrumentation/libnanox.la
instrumentation_libnanox_chpl_la_SOURCES=$(chpl_sources)
instrumentation_libnanox_chpl_la_DEPENDENCIES=instrumentation/libnanox.la
endif

if is_instrumentation_debug_enabled
instrumentation_debug_LTLIBRARIES+=instrumentation-debug/libnanox-chpl.la

instrumentation_debug_libnanox_chpl_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) $(chpl_flags)
instrumentation_debug_libnanox_chpl_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_chpl_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
instrumentation_debug_libnanox_chpl_la_LIBADD=instrumentation-debug/libnanox.la
instrumentation_debug_libnanox_chpl_la_SOURCES=$(chpl_sources)
instrumentation_debug_libnanox_chpl_la_DEPENDENCIES=instrumentation-debug/libnanox.la
endif

if is_performance_enabled
performance_LTLIBRARIES+=performance/libnanox-chpl.la

performance_libnanox_chpl_la_CPPFLAGS=$(common_performance_CPPFLAGS) $(chpl_flags)
performance_libnanox_chpl_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_chpl_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags)
performance_libnanox_chpl_la_LIBADD=performance/libnanox.la
performance_libnanox_chpl_la_SOURCES=$(chpl_sources)
performance_libnanox_chpl_la_DEPENDENCIES=performance/libnanox.la
endif
endif

###################################################
