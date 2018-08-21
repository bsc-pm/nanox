#####################################################################################
#      Copyright 2009-2018 Barcelona Supercomputing Center                          #
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
#      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            #
#####################################################################################

include $(top_srcdir)/src/common.am

pkginclude_HEADERS=\
	nanos-mpi.h \
	$(END)

devinclude_HEADERS = \
	mpiprocessor_fwd.hpp \
	mpiprocessor_decl.hpp \
	mpiprocessor.hpp \
	mpiremotenode_fwd.hpp \
	mpiremotenode_decl.hpp \
	mpiremotenode.hpp \
	mpidevice.hpp \
	mpidd.hpp \
	mpithread_fwd.hpp \
	mpithread.hpp \
	hostinfo.hpp \
	info.hpp \
	request.hpp \
	status.hpp \
	$(END)


offload_sources= \
 nanos-mpi.h \
 mpiall.cpp \
 mpi-api.cpp \
 mpiprocessor_fwd.hpp \
 mpiprocessor_decl.hpp \
 mpiprocessor.hpp \
 mpiprocessor.cpp \
 mpiremotenode_fwd.hpp \
 mpiremotenode_decl.hpp \
 mpiremotenode.hpp \
 mpiremotenode.cpp \
 mpidevice.hpp \
 mpidevice.cpp \
 mpidd.hpp \
 mpidd.cpp \
 mpithread_fwd.hpp \
 mpithread.hpp \
 mpithread.cpp \
 hostinfo.hpp \
 hostinfo.cpp \
 info.hpp \
 request.hpp \
 status.hpp \
 pagealignedallocator_decl.hpp \
 pagealignedallocator.hpp \
 mpiworker.hpp \
 mpispawn_fwd.hpp \
 mpispawn.hpp \
 rmi/allocate.hpp \
 rmi/cachecommand.cpp \
 rmi/cachecommand.hpp \
 rmi/cachepayload.hpp \
 rmi/command.cpp \
 rmi/command.hpp \
 rmi/commandid.hpp \
 rmi/commandpayload.hpp \
 rmi/commanddispatcher_decl.hpp \
 rmi/commanddispatcher.hpp \
 rmi/commandchannel.hpp \
 rmi/commandrequestor.hpp \
 rmi/commandservant.hpp \
 rmi/copydevtodev.hpp \
 rmi/copyin.hpp \
 rmi/copyout.hpp \
 rmi/createauxthread.hpp \
 rmi/finish.hpp \
 rmi/free.hpp \
 rmi/init.hpp \
 rmi/realloc.hpp \
 $(END)

pe_mpi_sources = \
 mpiplugin.hpp \
 mpiplugin.cpp \
 $(END)

EXTRA_DIST=$(offload_sources) $(pe_mpi_sources)


noinst_LTLIBRARIES =
debug_LTLIBRARIES =
instrumentation_LTLIBRARIES =
instrumentation_debug_LTLIBRARIES =
performance_LTLIBRARIES =

mpioffloaddir = $(prefix)/src/arch/mpi

mpioffload_DATA=\
 $(offload_sources)\
 $(pe_mpi_sources)\
 nanos-config.h \
 nanox_compiler \
 test_flags \
 $(END)

if is_debug_enabled
mpioffload_DATA+= debug_flags
endif
if is_instrumentation_enabled
mpioffload_DATA+= instrumentation_flags
endif
if is_instrumentation_debug_enabled
mpioffload_DATA+= instrumentation_debug_flags
endif
if is_performance_enabled
mpioffload_DATA+= performance_flags
endif

test_cxxflags = $(common_includes) $(bin_cxxflags)

.PHONY: test_flags debug_flags instrumentation_flags instrumentation_debug_flags performance_flags nanos-config.h

clean-local:
	rm -f test_flags debug_flags instrumentation_flags instrumentation_debug_flags performance_flags nanos-config.h

test_flags:
	echo -n "$(test_cxxflags) -I$(devincludedir)" | sed "s@$(top_srcdir)@$(abs_top_srcdir)@g" > $@

debug_flags:
	echo -n "$(common_debug_CPPFLAGS) -DNANOX_PREFIX=\"$(prefix)\" $(common_debug_CXXFLAGS)" > $@
	sed -i "s/-I[^\ ]*//g; s/config.h/nanos-config.h/" $@
	echo -n " -I$(devincludedir) -I$(mpioffloaddir)" >> $@

instrumentation_flags:
	echo -n "$(common_instrumentation_CPPFLAGS) -DNANOX_PREFIX=\"$(prefix)\" $(common_instrumentation_CXXFLAGS)" > $@
	sed -i "s/-I[^\ ]*//g; s/config.h/nanos-config.h/" $@
	echo -n " -I$(devincludedir) -I$(mpioffloaddir)" >> $@

instrumentation_debug_flags:
	echo -n "$(common_instrumentation_debug_CPPFLAGS) -DNANOX_PREFIX=\"$(prefix)\" $(common_instrumentation_debug_CXXFLAGS)" > $@
	sed -i "s/-I[^\ ]*//g; s/config.h/nanos-config.h/" $@
	echo -n " -I$(devincludedir) -I$(mpioffloaddir)" >> $@

performance_flags:
	echo -n "$(common_performance_CPPFLAGS) -DNANOX_PREFIX=\"$(prefix)\" $(common_performance_CXXFLAGS)" > $@
	sed -i "s/-I[^\ ]*//g; s/config.h/nanos-config.h/" $@
	echo -n " -I$(devincludedir) -I$(mpioffloaddir)" >> $@

nanos-config.h: $(abs_top_builddir)/config.h
	cp $(abs_top_builddir)/config.h $@

