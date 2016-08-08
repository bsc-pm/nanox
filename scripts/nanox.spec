# Good tutorials:
# http://freshrpms.net/docs/fight/
# http://rpm.org/api/4.4.2.2/conditionalbuilds.html
# http://fedoraproject.org/wiki/How_to_create_an_RPM_package
# http://backreference.org/2011/09/17/some-tips-on-rpm-conditional-macros/
# Repos:
# http://eureka.ykyuen.info/2010/01/06/opensuse-create-your-own-software-repository-1/
# http://en.opensuse.org/SDB:Creating_YaST_installation_sources

# Build line:
# rpmbuild -v -bb --clean SPECS/nanox.spec -with extrae && cp RPMS/x86_64/* ~/source/rpm_repo/x86_64/ && createrepo ~/source/rpm_repo

%{?_with_extrae: %define _with_extrae 1}
%{!?_with_extrae: %define _with_extrae 0}
%define _name        nanox

%if 0%{?suse_version}
%define distro       opensuse%{?suse_version}
%else
%define distro       %{?dist}
%endif

%define buildroot    %{_topdir}/%{_name}-%{_version}-root
# Avoid "*** ERROR: No build ID note found in XXXXXXX"
%global debug_package   %{nil}

# Override prefix if _rpm_prefix is given
%{?_rpm_prefix: %define _prefix  %{_rpm_prefix} }
# Override distribution flags
%define configure ./configure --host=%{_host} --build=%{_build} \\\
        --program-prefix=%{?_program_prefix} \\\
        --prefix=%{_prefix} \\\
        --exec-prefix=%{_exec_prefix} \\\
        --bindir=%{_bindir} \\\
        --sbindir=%{_sbindir} \\\
        --sysconfdir=%{_sysconfdir} \\\
        --datadir=%{_datadir} \\\
        --includedir=%{_includedir} \\\
        --libdir=%{_libdir} \\\
        --libexecdir=%{_libexecdir} \\\
        --localstatedir=%{_localstatedir} \\\
        --sharedstatedir=%{_sharedstatedir} \\\
        --mandir=%{_mandir} \\\
        --infodir=%{_infodir}

BuildRoot:     %{buildroot}
Summary:       Nanos++
License:       GPL
%if %_with_extrae
Name:          %{_name}-extrae
%else
Name:          %{_name}-no-extrae
%endif
Version:       %{_version}
Release:       %{_release}%{distro}
Source:        %{_name}-%{_version}.tar.gz
Prefix:        %{_prefix}
Group:         Development/Tools
Provides:      %{_name}
%if %_with_extrae
BuildRequires: extrae
Requires:      extrae
Conflicts:     %{_name}-no-extrae
%else
Conflicts:     %{_name}-extrae
%endif

%if %_with_extrae
%description
Nanos++ with extrae support.
%else
%description
Nanos++ without extrae support.
%endif

%prep
%setup -q -n %{_name}-%{_version}

%build
%if %_with_extrae
%configure --with-extrae=%{_prefix}
%else
%configure
%endif
make -j%{threads}

#%check
#make check

%install
make install DESTDIR=%{buildroot}

%files
%defattr(-,root,root)
%{_bindir}/*
%{_libdir}/debug/*
%{_libdir}/instrumentation/*
#%{_libdir}/instrumentation-debug/*
%{_libdir}/performance/*
%{_includedir}/*
%{_datarootdir}/doc/nanox/*
