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

SUBDIRS = . src tests doc scripts
DIST_SUBDIRS = tests doc src scripts
EXTRA_DIST = ChangeLog

ACLOCAL_AMFLAGS=-I m4

rpm: dist-gzip
	$(MAKE) -C scripts rpm

deb: dist-gzip
	$(MAKE) -C scripts deb

dist-hook:
	if [ -x "$(GIT)" ]; \
	then \
	    "$(GIT)" --git-dir=$(top_srcdir)/.git log --pretty=format:"[%h] %cd : %s (%an)" --date=short > $(distdir)/ChangeLog; \
	    git_run_version=`"$(GIT)" --git-dir=$(top_srcdir)/.git show --pretty=format:"%h %ci" HEAD | head -n 1`; \
	    git_run_branch=`"$(GIT)" --git-dir=$(top_srcdir)/.git branch | grep ^* | sed s/*\ //g`; \
	    git_version="git $${git_run_branch} $${git_run_version} developer version"; \
	    echo $${git_version} > ${distdir}/VERSION; \
	elif [ -e $(top_srcdir)/VERSION ]; \
	then \
	    cp $(top_srcdir)/VERSION $(distdir)/VERSION; \
	fi
