##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

# This directory is only listed in DIST_SUBDIRS, not SUBDIRS, so its contents
# will not be built by default, but it will participate in "make distclean" and
# friends.

# override the normal compilers for the tests
CC = @TEST_CC@
F77 = @TEST_F77@

LDADD = $(top_builddir)/libromio.la

AM_CPPFLAGS = $(ROMIO_INCLUDE)
AM_CPPFLAGS += -I$(top_builddir)/include -I$(top_srcdir)/include
AM_CPPFLAGS += -I$(top_builddir)/adio/include -I$(top_srcdir)/adio/include @mpl_includedir@
AM_CFLAGS = $(USER_CFLAGS)
AM_FFLAGS = $(USER_FFLAGS)

CTESTS = simple async coll_test misc file_info excl \
     atomicity noncontig i_noncontig noncontig_coll split_coll shared_fp \
     psimple error status noncontig_coll2 aggregation1 aggregation2 \
     async-multiple ordered_fp hindexed types_with_zeros darray_read syshints \
     fstype_prefix ghost_cell check_type_contig buf_noncontig pio_noncontig \
     large_dtype atomicity_coll ntimes_filetype ntimes_buftype \
     btype_lt_ftype resize_buftype nvars indexed_fsize hindexed_fsize \
     struct_fsize

SKIP_TESTS = perf coll_perf large_array large_file external32

FTESTS = fcoll_test fperf fmisc pfcoll_test

check_PROGRAMS = $(CTESTS) $(SKIP_TESTS)
TESTS = $(CTESTS)

if BUILD_F77_TESTS

check_PROGRAMS += $(FTESTS)
TESTS += $(FTESTS)
nodist_fperf_SOURCES = fperf.f
nodist_fcoll_test_SOURCES = fcoll_test.f
nodist_fmisc_SOURCES = fmisc.f
nodist_pfcoll_test_SOURCES = pfcoll_test.f

endif BUILD_F77_TESTS

TESTS_ENVIRONMENT = export srcdir="$(srcdir)";

TEST_EXTENSIONS = .sh
LOG_COMPILER = $(builddir)/parallel_run.sh
SH_LOG_COMPILER =

EXTRA_DIST = test_hintfile

# build check targets but not invoke
tests-local: all $(check_PROGRAMS)

