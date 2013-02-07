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

SUBDIRS = .

EXTRA_DIST=\
           astyle-nanox.sh \
           bets \
           headache.cfg \
           headerfy.sh \
           Makefile.am \
           preamble.txt \
           extrae_post_process.sh \
           nanox.spec \
           nanox\
           $(END)

bin_SCRIPTS=\
	extrae_post_process.sh\
   nanox\
	$(END)

rpm:
	@echo "Generating RPM structure"
	mkdir -p SOURCES BUILD RPMS SPECS SRPMS
	cp $(top_build_prefix)*.tar.gz SOURCES
	rpmbuild -v -bb --clean $(srcdir)/nanox.spec --define '_topdir $(CURDIR)' --define 'version $(VERSION)' --without extrae

clean-rpm:
	-rm -rf BUILDROOT SOURCES BUILD RPMS SPECS SRPMS


deb:
	cp $(top_build_prefix)*.tar.gz nanox_$(VERSION).orig.tar.gz
	tar -xf nanox_$(VERSION).orig.tar.gz
	mv nanox-$(VERSION) upstream_dir
	cp -rf $(srcdir)/debian upstream_dir/
	cp -rf upstream_dir/COPYING upstream_dir/debian/copyright
	-rm -f upstream_dir/debian/changelog

# Generate an empty changelog if git is not present
	if test ! -d $(top_srcdir)/.git; \
	then dch --create -c upstream_dir/debian/changelog --empty --package nanox -v $(VERSION) "$(VERSION)"; \
	else cd $(srcdir); git log -1 --pretty=format:"nanox ($(VERSION)) UNRELEASED; urgency=low%x0A%x0A  * %h %s%x0A%x0A -- %an <%ae>  %aD" >changelog; \
	fi
# I don't see another way of doing this
	if test -d $(top_srcdir)/.git; \
	then mv -f $(srcdir)/changelog upstream_dir/debian/changelog; \
	fi

	cd upstream_dir; DEB_BUILD_OPTIONS="nocheck" debuild -us -uc
	-rm -rf upstream_dir

clean-deb:
	-rm -f nanox_$(VERSION).orig.tar.gz nanox_$(VERSION).debian.tar.gz nanox_$(VERSION).dsc nanox_$(VERSION)*.build nanox_$(VERSION)*.changes *.deb
	-rm -rf upstream_dir

clean-local: clean-rpm clean-deb
