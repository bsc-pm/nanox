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

EXTRA_DIST = api-generator.in core-generator.in gpu-generator.in mixed-generator.in nanox-creation-generator.in api-omp-generator.in mcc-openmp-generator.in mcc-ompss-generator.in resiliency-generator.in config.py

noinst_SCRIPTS = api-generator core-generator gpu-generator mixed-generator nanox-creation-generator api-omp-generator mcc-openmp-generator mcc-ompss-generator resiliency-generator

CLEANFILES = api-generator core-generator gpu-generator mixed-generator nanox-creation-generator api-omp-generator mcc-openmp-generator mcc-ompss-generator resiliency-generator

all:
	chmod 755 api-generator
	chmod 755 core-generator
	chmod 755 gpu-generator
	chmod 755 mixed-generator
	chmod 755 nanox-creation-generator
	chmod 755 api-omp-generator
	chmod 755 mcc-openmp-generator
	chmod 755 mcc-ompss-generator
	chmod 755 resiliency-generator