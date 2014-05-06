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
	mpidevice_decl.hpp \
	mpidevice.hpp \
	mpidd.hpp \
	mpithread.hpp \
	$(END)

mpi_api_sources = \
	nanos-mpi.h \
	mpi-api.cpp \
	$(END)

mpi_sources = \
	mpiprocessor_fwd.hpp \
	mpiprocessor_decl.hpp \
	mpiprocessor.hpp \
	mpiprocessor.cpp \
	mpiremotenode_fwd.hpp \
	mpiremotenode_decl.hpp \
	mpiremotenode.hpp \
	mpiremotenode.cpp \
	mpidevice_decl.hpp \
	mpidevice.hpp \
	mpidevice.cpp \
	mpidd.hpp \
	mpidd.cpp \
	mpithread.hpp \
	mpithread.cpp \
	$(END)

pe_mpi_sources = \
	mpiplugin.cpp \
	$(END)

mpi_api_flags = -I$(top_srcdir)/src/apis/c \
                -DNANOX_PREFIX=\"$(NANOX_MPI_PREFIX)\"

noinst_LTLIBRARIES =
debug_LTLIBRARIES =
instrumentation_LTLIBRARIES =
instrumentation_debug_LTLIBRARIES =
performance_LTLIBRARIES = 


install-exec-local:

	command -v rsync  >/dev/null && rsync -az --include=*/ --include=*.h --include=*.hpp --exclude=* $(top_srcdir)/src/ $(NANOX_MPI_PREFIX)/src || (echo "rsync not found, using cp" ; cp -R $(top_srcdir)/src/ $(NANOX_MPI_PREFIX)/)
	cp -R $(top_srcdir)/src/arch/mpi $(NANOX_MPI_PREFIX)/src/arch/
	chmod 755 $(top_srcdir)/src 

if DEBUG_VERSION
nanos_debug_cxxflags=$(common_includes) $(common_debug_cxxflags) ${mpi_api_flags}
	echo ${nanos_debug_cxxflags} | sed "s@${top_srcdir}@${NANOX_MPI_PREFIX}@g" > $(NANOX_MPI_PREFIX)/src/arch/mpi/debug_flags
endif

if INSTRUMENTATION_VERSION
nanos_instr_cxxflags=$(common_includes) $(common_instr_cxxflags) ${mpi_api_flags}
	echo ${nanos_instr_cxxflags} | sed "s@${top_srcdir}@${NANOX_MPI_PREFIX}@g" > $(NANOX_MPI_PREFIX)/src/arch/mpi/instrumentation_flags
endif

if INSTRUMENTATION_DEBUG_VERSION
nanos_instrdebug_cxxflags=$(common_includes) $(common_instrdebug_cxxflags) ${mpi_api_flags}
	echo ${nanos_instrdebug_cxxflags} | sed "s@${top_srcdir}@${NANOX_MPI_PREFIX}@g" > $(NANOX_MPI_PREFIX)/src/arch/mpi/instrumentation_debug_flags
endif


if PERFORMANCE_VERSION
nanos_perfo_cxxflags=$(common_includes) $(common_perfo_cxxflags) ${mpi_api_flags}
	echo ${nanos_perfo_cxxflags} | sed "s@${top_srcdir}@${NANOX_MPI_PREFIX}@g" > $(NANOX_MPI_PREFIX)/src/arch/mpi/performance_flags 
endif