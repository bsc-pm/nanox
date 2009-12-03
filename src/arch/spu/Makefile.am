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

noinst_LTLIBRARIES = libspu.la

#.spu.o: $<

#/usr/bin/spu-gcc        -W -Wall -Winline  -I.  -I /opt/cell/sdk/usr/spu/include  -fstack-check -O3 -c hello_world_spu.c
#/usr/bin/spu-gcc -o hello_world_spu  hello_world_spu.o      -Wl,-N
#/usr/bin/ppu-embedspu -m32 hello_world_spu hello_world_spu hello_world_spu-embed.o
#/usr/bin/ppu-ar -qcs libhello_world_spu.a hello_world_spu-embed.o

spu_idle_embed.o : spu_idle.spu
	spu-cc -I /opt/cell/sdk/usr/spu/include -x c -o spu_idle.o $<
	ppu-embedspu  -m32 spu_idle spu_idle.o $@

libspu_la_CXXFLAGS=$(common_includes) $(common_cxxflags) -I/opt/cell/sdk/usr/include
libspu_la_SOURCES = \
                spudd.hpp \
                spudd.cpp \
		spuprocessor.hpp \
		spuprocessor.cpp \
		sputhread.hpp \
		sputhread.cpp \
                $(END)

libspu_la_LIBADD = \
		spu_idle_embed.o \
		$(END)


