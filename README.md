Nanos++ Runtime Library
=======================

Nanos++ is a runtime library designed to serve as runtime support in parallel
environments. It is mainly used to support **OmpSs** (an extension to the
OpenMP programming model based only in tasks). The runtime is developed at the
**Barcelona Supercomputing Center** within the **Programming Models** group.
Appart from OmpSs Nanos++ also supports most of the  **OpenMP** 3.1 features
and includes some additional extensions (some of them also introduced in
following OpenMP releases).

The runtime provides several services to support task parallelism using a
synchronization mechanism based on **data-dependencies**. Data parallelism is
also supported by means of services mapped on top of its task support. Task are
implemented as **user-level threads** when possible (currently x86, x86-64,
ia64, arm, ppc32 and ppc64 are supported). It also provides support for
maintaining coherence across different address spaces (such as with GPUs or
cluster nodes) by means of a **directory/cache** mechanism.

The main purpose of Nanos++ RTL is to be used in research of parallel
programming environments. Our aim has been to enable easy development of
different parts of the runtime so researchers have a platform that allows them
to try different mechanisms. As such it is designed to be **extensible by means
of plugins**.  The scheduling policy, the throttling policy, the dependence
approach, the barrier implementations, slicers and worksharing mechanisms, the
instrumentation layer and the architectural dependant level are examples of
plugins that developers may easily implements using Nanos++. This extensibility
does not come for free. The runtime overheads are slightly increased, but there
should be low enough for results to be meaningful except for cases of
extreme-fine grain applications.

You can find further information about the Nanos++ RTL usage in our
[user guide](https://pm.bsc.es/ompss-docs/user-guide)
and about Nanos++ RTL development in our
[developers guide](doc/developers_guide.md).

Reporting New Issues
-------------------

Should you find a bug or want to make a feature request you can create a new
ticket. As Nanos++ is a medium-sized piece of software so, in order to make bug
tracking as useful as possible, you may want to read these guidelines:

  1. Before reporting an issue **spend some time checking existing ones**. Maybe
your problem has already been reported and a fix is ongoing or planned.
Duplicated issues will be resolved as duplicated and no further action will be
taken on them. Sometimes it is not obvious what is duplicated or not and you
will have serious doubts about it. In this case just create the issue and add
references to those issues you believe related with in its description.

  2. Please, try to **make a useful report**. Where useful means we can figure out
the source of your problem. A gdb backtrace it can be useful to use, be sure to
use the debug version of Nanos++ in order to obtain an accurate backtrace. The
debug version can be enabled passing the --debug flag to Mercurium when the
application is built.

  3. Use the CC field to stay tuned to changes in a bug. Add your username or
e-mail there to track a bug and receive all the notifications due the different
actions taken in this ticket.

If you have any questions or suggestions you can contact the programming model
group at BSC by sending an e-mail to **pm-tools**\*.  You can also join our
**pm-tools-users** mailing list by sending an e-mail to **pm-tools-users-join**\*.

***

(\*) All our email accounts are hosted at **bsc.es** (i.e. \<account-name\>@bsc.es).

