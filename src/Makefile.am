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

SUBDIRS = arch support plugins core apis pms utils

EXTRA_DIST = plugins-list.in

noinst_DATA = plugins-list

CLEANFILES = plugins-list

plugins-list: plugins-list.in
	rm -f plugins_list
	tmp_builddir=$(abs_top_builddir) ; \
	sed \
		-e's|\(.*\)|'"$$tmp_builddir"'/src/\1/.libs|g' \
		$(srcdir)/plugins-list.in > plugins-list.tmp
	sed \
		-e :a -e N -e's/\n/:/' -e ta \
		plugins-list.tmp > plugins-list
	rm -f plugins-list.tmp

