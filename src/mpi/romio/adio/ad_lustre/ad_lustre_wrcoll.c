/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "ad_lustre.h"
#include "adio_extern.h"


#ifdef HAVE_LUSTRE_LOCKAHEAD
/* in ad_lustre_lock.c */
void ADIOI_LUSTRE_lock_ahead_ioctl(ADIO_File fd,
                                   int avail_cb_nodes, ADIO_Offset next_offset, int *error_code);

/* Handle lock ahead.  If this write is outside our locked region, lock it now */
#define ADIOI_LUSTRE_WR_LOCK_AHEAD(fd,cb_nodes,offset,error_code)           \
if (fd->hints->fs_hints.lustre.lock_ahead_write) {                          \
    if (offset > fd->hints->fs_hints.lustre.lock_ahead_end_extent) {        \
        ADIOI_LUSTRE_lock_ahead_ioctl(fd, cb_nodes, offset, error_code);    \
    }                                                                       \
    else if (offset < fd->hints->fs_hints.lustre.lock_ahead_start_extent) { \
        ADIOI_LUSTRE_lock_ahead_ioctl(fd, cb_nodes, offset, error_code);    \
    }                                                                       \
}
#else
#define ADIOI_LUSTRE_WR_LOCK_AHEAD(fd,cb_nodes,offset,error_code)
#endif


/* prototypes of functions used for collective writes only. */
static void ADIOI_LUSTRE_Exch_and_write(ADIO_File fd, const void *buf,
                                        MPI_Datatype buftype,
                                        ADIOI_Flatlist_node *flat_buf,
                                        int nprocs, int myrank,
                                        ADIOI_Access *others_req,
                                        ADIOI_Access *my_req,
                                        ADIO_Offset *offset_list,
                                        ADIO_Offset *len_list,
                                        ADIO_Offset min_st_loc, ADIO_Offset max_end_loc,
                                        MPI_Count contig_access_count,
                                        ADIO_Offset **buf_idx, int *error_code);
static void ADIOI_LUSTRE_Fill_send_buffer(ADIO_File fd, const void *buf,
                                          ADIOI_Flatlist_node *flat_buf,
                                          char **send_buf,
                                          ADIO_Offset *offset_list,
                                          ADIO_Offset *len_list, MPI_Count *send_size,
                                          MPI_Request *requests,
                                          MPI_Count *sent_to_proc, int nprocs,
                                          int myrank, MPI_Count contig_access_count,
                                          ADIO_Offset *send_buf_idx,
                                          MPI_Count *curr_to_proc,
                                          MPI_Count *done_to_proc, int iter, MPI_Aint buftype_extent);
static void ADIOI_LUSTRE_W_Exchange_data(ADIO_File fd, const void *buf,
                                         char *write_buf,
                                         ADIOI_Flatlist_node *flat_buf,
                                         ADIO_Offset *offset_list,
                                         ADIO_Offset *len_list, MPI_Count *send_size,
                                         MPI_Count *recv_size, ADIO_Offset range_off,
                                         MPI_Count range_size, MPI_Count *count,
                                         MPI_Count *start_pos,
                                         MPI_Count *sent_to_proc, int nprocs,
                                         int myrank,
                                         MPI_Count contig_access_count,
                                         ADIOI_Access *others_req,
                                         ADIO_Offset *send_buf_idx,
                                         MPI_Count *curr_to_proc,
                                         MPI_Count *done_to_proc,
                                         int iter, MPI_Aint buftype_extent,
                                         ADIO_Offset *buf_idx,
                                         ADIO_Offset **srt_off, MPI_Count **srt_len, MPI_Count *srt_num,
                                         int *error_code);
static void ADIOI_LUSTRE_IterateOneSided(ADIO_File fd, const void *buf,
                                         ADIO_Offset * offset_list, ADIO_Offset * len_list,
                                         MPI_Count contig_access_count,
                                         MPI_Count currentValidDataIndex, MPI_Aint count,
                                         int file_ptr_type, ADIO_Offset offset,
                                         ADIO_Offset start_offset, ADIO_Offset end_offset,
                                         ADIO_Offset firstFileOffset, ADIO_Offset lastFileOffset,
                                         MPI_Datatype buftype, int myrank, int *error_code);

/* ADIOI_LUSTRE_Calc_my_req() - calculate what portions of the access requests
 * of this process are located in the file domains of I/O aggregators.
 */
static
void ADIOI_LUSTRE_Calc_my_req(ADIO_File fd, ADIO_Offset *offset_list,
                              ADIO_Offset *len_list, MPI_Count contig_access_count,
                              int nprocs,
                              int *count_my_req_procs_ptr,
                              MPI_Count **count_my_req_per_proc_ptr,
                              ADIOI_Access **my_req_ptr, int buftype_is_contig,
                              ADIO_Offset **buf_idx)
{
    int i, proc, *aggr_ranks, count_my_req_procs;
    MPI_Count l, *count_my_req_per_proc;
    size_t nelems;
    ADIO_Offset avail_len, rem_len, curr_idx, off, *ptr;
    ADIO_Offset *avail_lens;
    ADIOI_Access *my_req;

    *count_my_req_per_proc_ptr = (MPI_Count *) ADIOI_Calloc(nprocs, sizeof(MPI_Count));
    count_my_req_per_proc = *count_my_req_per_proc_ptr;
    /* count_my_req_per_proc[i] gives the number of contiguous requests of this
     * process that fall in process i's file domain.
     */

    /* First pass just to calculate how much space to allocate for my_req.
     * contig_access_count has been calculated way back in
     * ADIOI_Calc_my_off_len()
     */
    aggr_ranks = (int *) ADIOI_Malloc(contig_access_count * sizeof(int));
    avail_lens = (ADIO_Offset *) ADIOI_Malloc(contig_access_count * sizeof(ADIO_Offset));

    /* nelems will be the number of offset-length pairs for my_req[] */
    nelems = 0;
    for (i = 0; i < contig_access_count; i++) {
        /* short circuit offset/len processing if len == 0
         * (zero-byte  read/write
         */
        /* short circuit offset/len processing if zero-byte read/write. */
        if (len_list[i] == 0)
            continue;

        off = offset_list[i];
        avail_len = len_list[i];
        /* ADIOI_LUSTRE_Calc_aggregator() modifies the value of avail_len to
         * return the amount that is only covered by the proc's file domain.
         * The remaining will still need to determine to whose file domain it
         * belongs. As ADIOI_LUSTRE_Calc_aggregator() can be expensive for
         * large value of contig_access_count, keep a copy of the returned
         * values proc and avail_len in aggr_ranks[] and avail_lens[] to be
         * used in the next for loop.
         */
        proc = ADIOI_LUSTRE_Calc_aggregator(fd, off, &avail_len);
        aggr_ranks[i] = proc;
        avail_lens[i] = avail_len;
        count_my_req_per_proc[proc]++;
        nelems++;

        /* rem_len is the amount of i's offset-length pair that is not covered
         * by proc's file domain.
         */
        rem_len = len_list[i] - avail_len;

        while (rem_len != 0) {
            off += avail_len;   /* point to first remaining byte */
            avail_len = rem_len;        /* save remaining size, pass to calc */
            proc = ADIOI_LUSTRE_Calc_aggregator(fd, off, &avail_len);
            count_my_req_per_proc[proc]++;
            nelems++;
            rem_len -= avail_len;       /* reduce remaining length by amount from fd */
        }
    }

    /* buf_idx is relevant only if buftype_is_contig.  buf_idx[i] gives the
     * starting index in user_buf where data will be sent to proc 'i'. This
     * allows sends to be done without extra buffer.
     */
    if (buf_idx != NULL && buftype_is_contig) {
        buf_idx[0] = (ADIO_Offset *) ADIOI_Malloc(nelems * sizeof(ADIO_Offset));
        for (i = 1; i < nprocs; i++)
            buf_idx[i] = buf_idx[i - 1] + count_my_req_per_proc[i - 1];
    }

    /* allocate space for my_req and its members offsets and lens */
    *my_req_ptr = (ADIOI_Access *) ADIOI_Malloc(nprocs * sizeof(ADIOI_Access));
    my_req = *my_req_ptr;
    my_req[0].offsets = (ADIO_Offset *) ADIOI_Malloc(nelems * 2 * sizeof(ADIO_Offset));

    count_my_req_procs = 0;
    ptr = my_req[0].offsets;
    for (i = 0; i < nprocs; i++) {
        if (count_my_req_per_proc[i]) {
            my_req[i].offsets = ptr;
            ptr += count_my_req_per_proc[i];
            my_req[i].lens = ptr;
            ptr += count_my_req_per_proc[i];
            count_my_req_procs++;
        }
        my_req[i].count = 0;    /* will be incremented where needed later */
    }

    /* now fill in my_req */
    curr_idx = 0;
    for (i = 0; i < contig_access_count; i++) {
        /* short circuit offset/len processing if zero-byte read/write. */
        if (len_list[i] == 0)
            continue;

        off = offset_list[i];
        proc = aggr_ranks[i];
        avail_len = avail_lens[i];

        l = my_req[proc].count;
        ADIOI_Assert(l < count_my_req_per_proc[proc]);
        if (buf_idx != NULL && buftype_is_contig) {
            buf_idx[proc][l] = curr_idx;
            curr_idx += avail_len;
        }
        rem_len = len_list[i] - avail_len;

        /* Each my_req[i] contains the number of this process's noncontiguous
         * requests that fall into process i's file domain. my_req[i].offsets[]
         * and my_req[i].lens store the offsets and lengths of the requests.
         */
        my_req[proc].offsets[l] = off;
        my_req[proc].lens[l] = avail_len;
        my_req[proc].count++;

        while (rem_len != 0) {
            off += avail_len;
            avail_len = rem_len;
            proc = ADIOI_LUSTRE_Calc_aggregator(fd, off, &avail_len);

            l = my_req[proc].count;
            ADIOI_Assert(l < count_my_req_per_proc[proc]);
            if (buf_idx != NULL && buftype_is_contig) {
                buf_idx[proc][l] = curr_idx;
                curr_idx += avail_len;
            }
            rem_len -= avail_len;

            my_req[proc].offsets[l] = off;
            my_req[proc].lens[l] = avail_len;
            my_req[proc].count++;
        }
    }
    ADIOI_Free(aggr_ranks);
    ADIOI_Free(avail_lens);

#ifdef AGG_DEBUG
    for (i = 0; i < nprocs; i++) {
        if (count_my_req_per_proc[i] > 0) {
            FPRINTF(stdout, "data needed from %d (count = %d):\n", i, my_req[i].count);
            for (l = 0; l < my_req[i].count; l++) {
                FPRINTF(stdout, "   off[%d] = %lld, len[%d] = %d\n",
                        l, (long long) my_req[i].offsets[l], l, (long long) my_req[i].lens[l]);
            }
        }
    }
#endif

    *count_my_req_procs_ptr = count_my_req_procs;
}

static
void ADIOI_LUSTRE_Calc_others_req(ADIO_File fd,
                                  int count_my_req_aggr,
                                  MPI_Count *count_my_req_per_aggr,
                                  ADIOI_Access *my_req,
                                  int nprocs,
                                  int myrank,
                                  int *count_others_req_procs_ptr,
                                  MPI_Count **count_others_req_per_proc_ptr,
                                  ADIOI_Access **others_req_ptr)
{
    /* determine what requests of other processes lie in this process's file
     * domain.
     *
     * count_others_req_procs = number of processes whose requests lie in this
     * process's file domain (including this process itself)
     * count_others_req_per_proc[i] indicates how many separate contiguous
     * requests of proc. i lie in this process's file domain.
     */

    int i, j, count_others_req_procs;
    MPI_Count *count_my_req_per_proc, *count_others_req_per_proc;
    MPI_Request *requests;
    ADIOI_Access *others_req;
    size_t memLen;
    ADIO_Offset *ptr;
    MPI_Aint *mem_ptrs;

/* first find out how much to send/recv and from/to whom */
#ifdef AGGREGATION_PROFILE
    MPE_Log_event(5026, 0, NULL);
#endif
    count_others_req_per_proc = (MPI_Count *) ADIOI_Malloc(nprocs * sizeof(MPI_Count));

    /* count_my_req_per_aggr[i]: the number of contiguous requests of this
     * process that fall in aggregator i's file domain.
     * count_others_req_per_proc[i] indicates how many noncontiguous requests
     * from process i that fall into this process's file domain.
     */
    count_my_req_per_proc = (MPI_Count*) ADIOI_Calloc(nprocs, sizeof(MPI_Count));
    for (i=0; i<fd->hints->cb_nodes; i++)
        count_my_req_per_proc[fd->hints->ranklist[i]] = count_my_req_per_aggr[i];

    MPI_Alltoall(count_my_req_per_proc, 1, MPI_COUNT,
                 count_others_req_per_proc, 1, MPI_COUNT, fd->comm);
    ADIOI_Free(count_my_req_per_proc);

    *others_req_ptr = (ADIOI_Access *) ADIOI_Malloc(nprocs * sizeof(ADIOI_Access));
    others_req = *others_req_ptr;

    memLen = 0;
    for (i = 0; i < nprocs; i++)
        memLen += count_others_req_per_proc[i];
    ptr = (ADIO_Offset *) ADIOI_Malloc(memLen * 2 * sizeof(ADIO_Offset));
    mem_ptrs = (MPI_Aint *) ADIOI_Malloc(memLen * sizeof(MPI_Aint));
    others_req[0].offsets = ptr;
    others_req[0].mem_ptrs = mem_ptrs;

    count_others_req_procs = 0;
    for (i = 0; i < nprocs; i++) {
        if (count_others_req_per_proc[i]) {
            others_req[i].count = count_others_req_per_proc[i];
            others_req[i].offsets = ptr;
            ptr += count_others_req_per_proc[i];
            others_req[i].lens = ptr;
            ptr += count_others_req_per_proc[i];
            others_req[i].mem_ptrs = mem_ptrs;
            mem_ptrs += count_others_req_per_proc[i];
            count_others_req_procs++;
        } else
            others_req[i].count = 0;
    }
    *count_others_req_per_proc_ptr = count_others_req_per_proc;

/* now send the calculated offsets and lengths to respective processes */

    requests = (MPI_Request *)
        ADIOI_Malloc(1 + (count_my_req_aggr + count_others_req_procs) * sizeof(MPI_Request));
/* +1 to avoid a 0-size malloc */

    j = 0;
    for (i = 0; i < nprocs; i++) {
        if (others_req[i].count == 0)
            continue;
        if (i == myrank)
            /* send to self uses memcpy(), here
             * others_req[i].count == my_req[myrank].count
             */
            memcpy(others_req[i].offsets, my_req[myrank].offsets,
                   2 * my_req[myrank].count * sizeof(ADIO_Offset));
        else
            MPI_Irecv(others_req[i].offsets, 2 * others_req[i].count,
                      ADIO_OFFSET, i, 0, fd->comm, &requests[j++]);
    }

    for (i = 0; i < fd->hints->cb_nodes; i++) {
        if (my_req[i].count && i != myrank)
            MPI_Isend(my_req[i].offsets, 2 * my_req[i].count,
                      ADIO_OFFSET, fd->hints->ranklist[i], 0, fd->comm,
                      &requests[j++]);
    }

    if (j) {
#ifdef MPI_STATUSES_IGNORE
        MPI_Waitall(j, requests, MPI_STATUSES_IGNORE);
#else
        MPI_Status *statuses = (MPI_Status *) ADIOI_Malloc(j * sizeof(MPI_Status));
        MPI_Waitall(j, requests, statuses);
        ADIOI_Free(statuses);
#endif
    }

    ADIOI_Free(requests);

    *count_others_req_procs_ptr = count_others_req_procs;
#ifdef AGGREGATION_PROFILE
    MPE_Log_event(5027, 0, NULL);
#endif
}

void ADIOI_LUSTRE_WriteStridedColl(ADIO_File fd, const void *buf, MPI_Aint count,
                                   MPI_Datatype buftype,
                                   int file_ptr_type, ADIO_Offset offset,
                                   ADIO_Status *status, int *error_code)
{
    /* Uses a generalized version of the extended two-phase method described in
     * "An Extended Two-Phase Method for Accessing Sections of Out-of-Core
     * Arrays", Rajeev Thakur and Alok Choudhary, Scientific Programming,
     * (5)4:301--317, Winter 1996.
     * http://www.mcs.anl.gov/home/thakur/ext2ph.ps
     */

    int i, nprocs, nonzero_nprocs, myrank, old_error, tmp_error;
    int do_collect = 0;
    MPI_Count contig_access_count = 0;
    int buftype_is_contig;
    ADIO_Offset orig_fp, start_offset, end_offset;
    ADIO_Offset min_st_loc = -1, max_end_loc = -1;
    ADIO_Offset *offset_list = NULL, *len_list = NULL;
    ADIOI_Flatlist_node *flat_buf = NULL;

    MPI_Comm_size(fd->comm, &nprocs);
    MPI_Comm_rank(fd->comm, &myrank);

    orig_fp = fd->fp_ind;

    /* Check if collective write is actually necessary, if cb_write hint isn't
     * disabled by users.
     */
    if (fd->hints->cb_write != ADIOI_HINT_DISABLE) {
        int is_interleaved;
        ADIO_Offset st_end[2], *st_end_all = NULL;

        /* Calculate and construct the list of starting file offsets and
         * lengths of write requests of this process. No inter-process
         * communication is needed.
         *
         * Note: end_offset points to the last byte-offset to be accessed.
         * e.g., if start_offset=0 and 100 bytes to be read, end_offset=99
         * No inter-process communication is needed. If this process has no
         * data to write, end_offset == (start_offset - 1)
         */
        ADIOI_Calc_my_off_len(fd, count, buftype, file_ptr_type, offset,
                              &offset_list, &len_list, &start_offset,
                              &end_offset, &contig_access_count);

        /* All processes gather starting and ending file offsets of requests
         * from all processes into st_end_all[]. Even indices of st_end_all[]
         * are start offsets, odd indices are end offsets. st_end_all[] is used
         * below to tell whether access across all process is interleaved.
         */
        st_end[0] = start_offset;
        st_end[1] = end_offset;
        st_end_all = (ADIO_Offset *) ADIOI_Malloc(nprocs * 2 * sizeof(ADIO_Offset));
        MPI_Allgather(st_end, 2, ADIO_OFFSET, st_end_all, 2, ADIO_OFFSET, fd->comm);

        /* Find the starting and ending file offsets of aggregate access region
         * and the number of processes that have non-zero length write
         * requests. Also, check whether accesses are interleaved across
         * processes. Below is a rudimentary check for interleaving, but should
         * suffice for the moment.
         */
        is_interleaved = 0;
        nonzero_nprocs = 0;
        for (i = 0; i < nprocs * 2; i += 2) {
            if (st_end_all[i] > st_end_all[i + 1]) {
                /* process rank (i/2) has no data to write */
                continue;
            }
            min_st_loc = st_end_all[i];
            max_end_loc = st_end_all[i + 1];
            nonzero_nprocs = 1;
            i += 2;
            break;
        }
        for (; i < nprocs * 2; i += 2) {
            if (st_end_all[i] > st_end_all[i + 1]) {
                /* process rank (i/2) has no data to write */
                continue;
            }
            if (st_end_all[i] < st_end_all[i - 1]) {
                /* start offset of process rank (i/2) is less than the end
                 * offset of process rank (i/2-1)
                 */
                is_interleaved = 1;
            }
            min_st_loc = MPL_MIN(st_end_all[i], min_st_loc);
            max_end_loc = MPL_MAX(st_end_all[i + 1], max_end_loc);
            nonzero_nprocs++;
        }
        ADIOI_Free(st_end_all);

        /* Two typical access patterns can benefit from collective write.
         *   1) access file regions among all processes are interleaved, and
         *   2) the individual request sizes are not too big, i.e. no bigger
         *      than hint coll_threshold.  Large individual requests may cause
         *      a high communication cost for redistributing requests to the
         *      I/O aggregators.
         */
        if (is_interleaved > 0) {
            do_collect = 1;
        } else {
            /* This ADIOI_LUSTRE_Docollect() calls MPI_Allreduce(), so all
             * processes must participate.
             */
            do_collect = ADIOI_LUSTRE_Docollect(fd, contig_access_count, len_list, nprocs);
        }
    }

    flat_buf = ADIOI_Flatten_and_find(buftype);
    if (flat_buf->count == 1) /* actually contiguous */
        buftype_is_contig = 1;
    else
        buftype_is_contig = 0;

    /* If collective I/O is not necessary, use independent I/O */
    if ((!do_collect && fd->hints->cb_write == ADIOI_HINT_AUTO) ||
        fd->hints->cb_write == ADIOI_HINT_DISABLE) {
        int filetype_is_contig;
        ADIOI_Flatlist_node *flat_ftype;

        if (offset_list != NULL)
            ADIOI_Free(offset_list);

        fd->fp_ind = orig_fp;
        flat_ftype = ADIOI_Flatten_and_find(fd->filetype);
        if (flat_ftype->count == 1) /* actually contiguous */
            filetype_is_contig = 1;
        else
            filetype_is_contig = 0;

        if (buftype_is_contig && filetype_is_contig) {
            ADIO_Offset off = 0;
            if (file_ptr_type == ADIO_EXPLICIT_OFFSET)
                off = fd->disp + offset * fd->etype_size;
            ADIO_WriteContig(fd, buf, count, buftype, file_ptr_type, off, status, error_code);
        } else {
            ADIO_WriteStrided(fd, buf, count, buftype, file_ptr_type, offset, status, error_code);
        }
        return;
    }

    if ((fd->romio_write_aggmethod == 1) || (fd->romio_write_aggmethod == 2)) {
        /* If user has set hint ROMIO_WRITE_AGGMETHOD env variable to to use a
         * one-sided aggregation method then do that at this point instead of
         * using the traditional MPI point-to-point communication, i.e.
         * MPI_Isend and MPI_Irecv.
         */
        ADIOI_LUSTRE_IterateOneSided(fd, buf, offset_list, len_list,
                                     contig_access_count, nonzero_nprocs, count,
                                     file_ptr_type, offset, start_offset, end_offset,
                                     min_st_loc, max_end_loc, buftype, myrank, error_code);
    } else {
        /* my_req[] is an array of nprocs access structures, one for each other
         * process whose file domain has this process's request */
        ADIOI_Access *my_req;
        int count_my_req_procs;
        MPI_Count *count_my_req_per_proc;

        /* others_req[] is an array of nprocs access structures, one for each
         * other process whose requests fall into this process's file domain,
         * i.e. is written by this process. */
        ADIOI_Access *others_req;
        int count_others_req_procs;
        MPI_Count *count_others_req_per_proc;
        ADIO_Offset **buf_idx = NULL;

        /* Calculate the portions of this process's write requests that fall
         * into the file domains of each I/O aggregator. No inter-process
         * communication is needed.
         */
        if (buftype_is_contig == 1)
            buf_idx = (ADIO_Offset **) ADIOI_Malloc(nprocs * sizeof(ADIO_Offset*));
        ADIOI_LUSTRE_Calc_my_req(fd, offset_list, len_list, contig_access_count,
                                 nprocs, &count_my_req_procs,
                                 &count_my_req_per_proc,
                                 &my_req, buftype_is_contig, buf_idx);

        /* Calculate the portions of all other process's requests fall into
         * this process's file domain (note only I/O aggregators are assigned
         * file domains). Inter-process communication is required to construct
         * others_req[], including MPI_Alltoall, MPI_Isend, MPI_Irecv, and
         * MPI_Waitall.
         *
         * count_others_req_procs is the number of processes whose requests
         * (including this process itself) fall into this process's file
         * domain.
         * count_others_req_per_proc[i] indicates how many noncontiguous
         * requests from process i that fall into this process's file domain.
         */
        ADIOI_LUSTRE_Calc_others_req(fd, count_my_req_procs,
                                     count_my_req_per_proc, my_req, nprocs,
                                     myrank,
                                     &count_others_req_procs,
                                     &count_others_req_per_proc, &others_req);

        /* Two-phase I/O: first communication phase to exchange write data from
         * all processes to the I/O aggregators, followed by the write phase
         * where only I/O aggregators write to the file. There is no collective
         * MPI communication in ADIOI_LUSTRE_Exch_and_write(), only MPI_Issend,
         * MPI_Irecv, and MPI_Waitall.
         */
        ADIOI_LUSTRE_Exch_and_write(fd, buf, buftype, flat_buf, nprocs, myrank,
                                    others_req, my_req, offset_list, len_list,
                                    min_st_loc, max_end_loc,
                                    contig_access_count, buf_idx,
                                    error_code);

        /* free all memory allocated */
        ADIOI_Free_others_req(nprocs, count_others_req_per_proc, others_req);

        ADIOI_Free(count_my_req_per_proc);
        if (buf_idx != NULL) {
            ADIOI_Free(buf_idx[0]);
            ADIOI_Free(buf_idx);
        }
        ADIOI_Free(my_req[0].offsets);
        ADIOI_Free(my_req);
    }
    ADIOI_Free(offset_list);

    /* If this collective write is followed by an independent write, it's
     * possible to have those subsequent writes on other processes race ahead
     * and sneak in before the read-modify-write completes.  We carry out a
     * collective communication at the end here so no one can start independent
     * I/O before collective I/O completes.
     *
     * need to do some gymnastics with the error codes so that if something
     * went wrong, all processes report error, but if a process has a more
     * specific error code, we can still have that process report the
     * additional information */
    old_error = *error_code;
    if (*error_code != MPI_SUCCESS)
        *error_code = MPI_ERR_IO;

    /* optimization: if only one process performing I/O, we can perform
     * a less-expensive Bcast. */
#ifdef ADIOI_MPE_LOGGING
    MPE_Log_event(ADIOI_MPE_postwrite_a, 0, NULL);
#endif
    if (fd->hints->cb_nodes == 1)
        MPI_Bcast(error_code, 1, MPI_INT, fd->hints->ranklist[0], fd->comm);
    else {
        tmp_error = *error_code;
        MPI_Allreduce(&tmp_error, error_code, 1, MPI_INT, MPI_MAX, fd->comm);
    }
#ifdef ADIOI_MPE_LOGGING
    MPE_Log_event(ADIOI_MPE_postwrite_b, 0, NULL);
#endif

    if ((old_error != MPI_SUCCESS) && (old_error != MPI_ERR_IO))
        *error_code = old_error;

#ifdef HAVE_STATUS_SET_BYTES
    if (status) {
        MPI_Count bufsize, size;
        /* Don't set status if it isn't needed */
        MPI_Type_size_x(buftype, &size);
        bufsize = size * count;
        MPIR_Status_set_bytes(status, buftype, bufsize);
    }
    /* This is a temporary way of filling in status. The right way is to
     * keep track of how much data was actually written during collective I/O.
     */
#endif

    fd->fp_sys_posn = -1;       /* set it to null. */
}

/* If successful, error_code is set to MPI_SUCCESS.  Otherwise an error code is
 * created and returned in error_code.
 */
static void ADIOI_LUSTRE_Exch_and_write(ADIO_File fd, const void *buf,
                                        MPI_Datatype buftype, ADIOI_Flatlist_node *flat_buf,
                                        int nprocs, int myrank, ADIOI_Access *others_req,
                                        ADIOI_Access *my_req,
                                        ADIO_Offset *offset_list,
                                        ADIO_Offset *len_list,
                                        ADIO_Offset min_st_loc, ADIO_Offset max_end_loc,
                                        MPI_Count contig_access_count,
                                        ADIO_Offset **buf_idx, int *error_code)
{
    /* Each process sends all its write requests to I/O aggregators based on
     * the file domain assignment to the aggregators. In this implementation,
     * a file is first divided into stripes and all its stripes are assigned to
     * I/O aggregators in a round-robin fashion. The collective write is
     * carried out in 'ntimes' rounds of two-phase I/O. Each round covers an
     * aggregate file region of size equal to the file stripe size times the
     * number of I/O aggregators. In other words, the 'collective buffer size'
     * used in each aggregator is always set equally to the file stripe size,
     * ignoring the MPI-IO hint 'cb_buffer_size'. There are other algorithms
     * allowing an aggregator to write more than a file stripe size in each
     * round, up to the cb_buffer_size hint. For those, refer to the paper:
     * Wei-keng Liao, and Alok Choudhary. "Dynamically Adapting File Domain
     * Partitioning Methods for Collective I/O Based on Underlying Parallel
     * File System Locking Protocols", in The Supercomputing Conference, 2008.
     */

    char *write_buf = NULL;
    int i;
    MPI_Count m, ntimes;
    MPI_Count *recv_curr_offlen_ptr, *recv_size, *recv_count, *recv_start_pos;
    MPI_Count *send_curr_offlen_ptr, *send_size, *sent_to_proc;
    MPI_Count *curr_to_proc, *done_to_proc, *srt_len = NULL, srt_num = 0;
    int stripe_size = fd->hints->striping_unit, avail_cb_nodes = fd->hints->cb_nodes;
    ADIO_Offset end_loc, req_off, iter_end_off, *off_list, step_size;
    ADIO_Offset *send_buf_idx, *this_buf_idx, *srt_off = NULL;
    MPI_Aint lb, buftype_extent;

    *error_code = MPI_SUCCESS;

    /* The aggregate access region (across all processes) of this collective
     * write starts from min_st_loc and ends at max_end_loc. The collective
     * write is carried out in 'ntimes' rounds of two-phase I/O. Each round
     * covers an aggregate file region of size 'step_size' written only by
     * 'avail_cb_nodes' number of processes (I/O aggregators). Note
     * non-aggregators must also participate all ntimes rounds to send their
     * requests to I/O aggregators.
     */

    /* step_size is the size of aggregate access region covered by each round
     * of two-phase I/O
     */
    step_size = ((ADIO_Offset) avail_cb_nodes) * stripe_size;

    /* align min_st_loc downward to the nearest file stripe boundary */
    min_st_loc -= min_st_loc % (ADIO_Offset) stripe_size;

    /* ntimes is the number of rounds of two-phase I/O */
    ntimes = (max_end_loc - min_st_loc + 1) / step_size;
    if ((max_end_loc - min_st_loc + 1) % step_size)
        ntimes++;

    /* off_list[m] is the starting file offset of this process's write region
     * in iteration m (file domain of iteration m). This offset may not be
     * aligned with file stripe boundaries. end_loc is the ending file offset
     * of this process's file domain.
     */
    off_list = (ADIO_Offset *) ADIOI_Malloc((ntimes + 2 * nprocs) * sizeof(ADIO_Offset));
    end_loc = -1;
    for (m = 0; m < ntimes; m++)
        off_list[m] = max_end_loc;
    for (i = 0; i < nprocs; i++) {
        for (MPI_Count j = 0; j < others_req[i].count; j++) {
            req_off = others_req[i].offsets[j];
            m = (int) ((req_off - min_st_loc) / step_size);
            off_list[m] = MPL_MIN(off_list[m], req_off);
            end_loc = MPL_MAX(end_loc, (others_req[i].offsets[j] + others_req[i].lens[j] - 1));
        }
    }

    /* end_loc >= 0 indicates this process has something to write.
     * Only I/O aggregators can have end_loc > 0. write_buf is the collective
     * buffer and only matter for I/O aggregators. It is allocated with space
     * of twice the file stripe size. The second half will be used to receive
     * write data from remote processes, which are later copied over to the
     * first half. Once communications are complete, the contents of first half
     * are written to file.
     */
    if (end_loc >= 0)
        write_buf = (char *) ADIOI_Malloc(stripe_size * 2);

    /* send_buf_idx and this_buf_idx are indices to user buffer for sending
     * this process's write data to remote aggregators. These two are used only
     * when user buffer is contiguous.
     */
    send_buf_idx = off_list + ntimes;
    if (flat_buf->count == 1)
        this_buf_idx = (ADIO_Offset *) ADIOI_Malloc(nprocs * sizeof(ADIO_Offset));

    /* allocate int buffers altogether at once in a single calloc call */
    recv_curr_offlen_ptr = (MPI_Count *) ADIOI_Calloc(nprocs * 9, sizeof(MPI_Count));
    send_curr_offlen_ptr = recv_curr_offlen_ptr + nprocs;
    /* their use is explained below. calloc initializes to 0. */

    recv_count = send_curr_offlen_ptr + nprocs;
    /* the number of off-len pairs to be received from each proc in an iteration. */

    send_size = recv_count + nprocs;
    /* array of data sizes to be sent to each proc in an iteration. */

    recv_size = send_size + nprocs;
    /* array of data sizes to be received from each proc in an iteration. */

    sent_to_proc = recv_size + nprocs;
    /* amount of data sent to each proc so far, initialized to 0 here. */

    curr_to_proc = sent_to_proc + nprocs;
    done_to_proc = curr_to_proc + nprocs;
    /* Above three are used in ADIOI_Fill_send_buffer only */

    recv_start_pos = done_to_proc + nprocs;
    /* used to store the starting value of recv_curr_offlen_ptr[i] in
     * this iteration */

    MPI_Type_get_extent(buftype, &lb, &buftype_extent);

    /* I need to check if there are any outstanding nonblocking writes to
     * the file, which could potentially interfere with the writes taking
     * place in this collective write call. Since this is not likely to be
     * common, let me do the simplest thing possible here: Each process
     * completes all pending nonblocking operations before completing.
     */
    /*ADIOI_Complete_async(error_code);
     * if (*error_code != MPI_SUCCESS) return;
     * MPI_Barrier(fd->comm);
     */

    /* min_st_loc has been downward aligned to the nearest file stripe
     * boundary, iter_end_off is the ending file offset of aggregate write
     * region of iteration m, upward aligned to the file stripe boundary.
     */
    iter_end_off = min_st_loc + step_size;

    for (m = 0; m < ntimes; m++) {
        int range_size;
        ADIO_Offset range_off;

        /* Note that MPI standard requires that displacements in filetypes are
         * in a monotonically nondecreasing order and that, for writes, the
         * filetypes cannot specify overlapping regions in the file. This
         * simplifies implementation a bit compared to reads.
         *
         * range_off     = starting file offset of this aggregator's write
         *                 region for this round (may not be aligned to stripe
         *                 boundary)
         * range_size    = size (in bytes) of this aggregator's write region
         *                 for this found
         * iter_end_off  = ending file offset of aggregate write region of this
         *                 round, and upward aligned to the file stripe
         *                 boundary. Note the aggregate write region of this
         *                 round starts from (iter_end_off-step_size) to
         *                 iter_end_off, aligned with file stripe boundaries.
         * send_size[i]  = total size in bytes of this process's write data
         *                 fall into process i's write region for this round.
         * recv_size[i]  = total size in bytes of write data to be received by
         *                 this process (aggregator) for this round.
         * recv_count[i] = the number of noncontiguous offset-length pairs from
         *                 process i fall into this aggregator's write region
         *                 for this round.
         */

        /* reset communication metadata to all 0s for this round */
        for (i = 0; i < nprocs; i++)
            recv_count[i] = send_size[i] = recv_size[i] = 0;

        range_off = off_list[m];
        range_size = (int) MPL_MIN(stripe_size - range_off % stripe_size, end_loc - range_off + 1);

        /* First calculate what should be communicated, by going through all
         * others_req and my_req to check which will be sent and received in
         * this round.
         */
        for (i = 0; i < nprocs; i++) {
            MPI_Count j;
            if (my_req[i].count) {
                if (flat_buf->count == 1)
                    this_buf_idx[i] = buf_idx[i][send_curr_offlen_ptr[i]];
                for (j = send_curr_offlen_ptr[i]; j < my_req[i].count; j++) {
                    if (my_req[i].offsets[j] < iter_end_off)
                        send_size[i] += my_req[i].lens[j];
                    else
                        break;
                }
                send_curr_offlen_ptr[i] = j;
            }
            if (others_req[i].count) {
                recv_start_pos[i] = recv_curr_offlen_ptr[i];
                for (j = recv_curr_offlen_ptr[i]; j < others_req[i].count; j++) {
                    if (others_req[i].offsets[j] < iter_end_off) {
                        recv_count[i]++;
                        others_req[i].mem_ptrs[j] =
                            (MPI_Count) (others_req[i].offsets[j] - range_off);
                        recv_size[i] += others_req[i].lens[j];
                    } else {
                        break;
                    }
                }
                recv_curr_offlen_ptr[i] = j;
            }
        }
        iter_end_off += step_size;

        /* redistribute (exchange) this process's write requests to I/O
         * aggregators. In ADIOI_LUSTRE_W_Exchange_data(), communication are
         * Issend and Irecv, but no collective communication.
         */
        ADIOI_LUSTRE_W_Exchange_data(fd, buf, write_buf, flat_buf, offset_list,
                                     len_list, send_size, recv_size, range_off, range_size,
                                     recv_count, recv_start_pos,
                                     sent_to_proc, nprocs, myrank,
                                     contig_access_count,
                                     others_req, send_buf_idx,
                                     curr_to_proc, done_to_proc, m,
                                     buftype_extent, this_buf_idx,
                                     &srt_off, &srt_len, &srt_num, error_code);

        if (*error_code != MPI_SUCCESS)
            goto over;

        /* if there is no data to write for this iteration m */
        if (srt_num == 0)
            continue;

        /* lock ahead the file starting from range_off */
        ADIOI_LUSTRE_WR_LOCK_AHEAD(fd, fd->hints->cb_nodes, range_off, error_code);
        if (*error_code != MPI_SUCCESS)
            goto over;

        /* When srt_num == 1, either there is no hole in the write buffer or
         * the file domain has been read into write buffer and updated with the
         * received write data. When srt_num > 1, holes have been found and the
         * list of sorted offset-length pairs describing noncontiguous writes
         * have been constructed. Call writes for each offset-length pair. Note
         * the offset-length pairs (represented by srt_off, srt_len, and
         * srt_num) have been coalesced in ADIOI_LUSTRE_W_Exchange_data().
         */
        for (i = 0; i < srt_num; i++) {
            MPI_Status status;

            /* all write requests in this round should fall into this range of
             * [range_off, range_off+range_size). This assertion should never fail.
             */
            ADIOI_Assert(srt_off[i] < range_off + range_size && srt_off[i] >= range_off);

            ADIO_WriteContig(fd, write_buf + (srt_off[i] - range_off), srt_len[i],
                             MPI_BYTE, ADIO_EXPLICIT_OFFSET, srt_off[i], &status, error_code);
            if (*error_code != MPI_SUCCESS)
                goto over;
        }
    }
  over:
    if (srt_off)
        ADIOI_Free(srt_off);
    if (srt_len)
        ADIOI_Free(srt_len);
    if (write_buf != NULL)
        ADIOI_Free(write_buf);
    ADIOI_Free(recv_curr_offlen_ptr);
    ADIOI_Free(off_list);
    if (flat_buf->count == 1)
        ADIOI_Free(this_buf_idx);
}

/* This subroutine is copied from ADIOI_Heap_merge(), but modified to coalesce
 * sorted offset-length pairs whenever possible.
 *
 * Heapify(a, i, heapsize); Algorithm from Cormen et al. pg. 143 modified for a
 * heap with smallest element at root. The recursion has been removed so that
 * there are no function calls. Function calls are too expensive.
 */
static
void heap_merge(const ADIOI_Access *others_req, const MPI_Count *count,
                ADIO_Offset *srt_off, MPI_Count *srt_len,
                const MPI_Count *start_pos, int nprocs, int nprocs_recv,
                MPI_Count *total_elements)
{
    typedef struct {
        ADIO_Offset *off_list;
        ADIO_Offset *len_list;
        MPI_Count nelem;
    } heap_struct;

    heap_struct *a, tmp;
    int i, j, heapsize, l, r, k, smallest;

    a = (heap_struct *) ADIOI_Malloc((nprocs_recv + 1) * sizeof(heap_struct));

    j = 0;
    for (i = 0; i < nprocs; i++) {
        if (count[i]) {
            a[j].off_list = others_req[i].offsets + start_pos[i];
            a[j].len_list = others_req[i].lens + start_pos[i];
            a[j].nelem = count[i];
            j++;
        }
    }

#define SWAP(x, y, tmp) { tmp = x ; x = y ; y = tmp ; }

    heapsize = nprocs_recv;

    /* Build a heap out of the first element from each list, with the smallest
     * element of the heap at the root. The first for loop is to find and move
     * the smallest a[*].off_list[0] to a[0].
     */
    for (i = heapsize / 2 - 1; i >= 0; i--) {
        k = i;
        for (;;) {
            r = 2 * (k + 1);
            l = r - 1;
            if ((l < heapsize) && (*(a[l].off_list) < *(a[k].off_list)))
                smallest = l;
            else
                smallest = k;

            if ((r < heapsize) && (*(a[r].off_list) < *(a[smallest].off_list)))
                smallest = r;

            if (smallest != k) {
                SWAP(a[k], a[smallest], tmp);
                k = smallest;
            } else
                break;
        }
    }

    /* The heap keeps the smallest element in its first element, i.e.
     * a[0].off_list[0].
     */
    j = 0;
    for (i = 0; i < *total_elements; i++) {
        /* extract smallest element from heap, i.e. the root */
        if (j == 0 || srt_off[j - 1] + srt_len[j - 1] < *(a[0].off_list)) {
            srt_off[j] = *(a[0].off_list);
            srt_len[j] = *(a[0].len_list);
            j++;
        } else {
            /* this offset-length pair can be coalesced into the previous one */
            srt_len[j - 1] = *(a[0].off_list) + *(a[0].len_list) - srt_off[j - 1];
        }
        (a[0].nelem)--;

        if (a[0].nelem) {
            (a[0].off_list)++;
            (a[0].len_list)++;
        } else {
            a[0] = a[heapsize - 1];
            heapsize--;
        }

        /* Heapify(a, 0, heapsize); */
        k = 0;
        for (;;) {
            r = 2 * (k + 1);
            l = r - 1;
            if ((l < heapsize) && (*(a[l].off_list) < *(a[k].off_list)))
                smallest = l;
            else
                smallest = k;

            if ((r < heapsize) && (*(a[r].off_list) < *(a[smallest].off_list)))
                smallest = r;

            if (smallest != k) {
                SWAP(a[k], a[smallest], tmp);
                k = smallest;
            } else
                break;
        }
    }
    ADIOI_Free(a);
    *total_elements = j;
}

static void ADIOI_LUSTRE_W_Exchange_data(ADIO_File fd, const void *buf,
                                         char *write_buf,
                                         ADIOI_Flatlist_node *flat_buf,
                                         ADIO_Offset *offset_list,
                                         ADIO_Offset *len_list, MPI_Count *send_size,
                                         MPI_Count *recv_size, ADIO_Offset range_off,
                                         MPI_Count range_size, MPI_Count *recv_count,
                                         MPI_Count *start_pos,
                                         MPI_Count *sent_to_proc, int nprocs,
                                         int myrank,
                                         MPI_Count contig_access_count,
                                         ADIOI_Access *others_req,
                                         ADIO_Offset *send_buf_idx,
                                         MPI_Count *curr_to_proc, MPI_Count *done_to_proc,
                                         int iter,
                                         MPI_Aint buftype_extent,
                                         ADIO_Offset *buf_idx,
                                         ADIO_Offset **srt_off, MPI_Count **srt_len, MPI_Count *srt_num,
                                         int *error_code)
{
    char *buf_ptr, *contig_buf, **send_buf = NULL;
    int i, nprocs_recv, nprocs_send, err;
    int nreqs, hole, check_hole;
    MPI_Count sum_recv;
    static size_t malloc_srt_num = 0;
    MPI_Request *requests;
    MPI_Status status;
    static char myname[] = "ADIOI_LUSTRE_W_EXCHANGE_DATA";

    /* calculate send receive metadata */
    *srt_num = 0;
    sum_recv = 0;
    nprocs_recv = 0;
    nprocs_send = 0;
    for (i = 0; i < nprocs; i++) {
        *srt_num += recv_count[i];
        sum_recv += recv_size[i];
        if (recv_size[i])
            nprocs_recv++;
        if (send_size[i])
            nprocs_send++;
    }

    /* determine whether checking holes is necessary */
    check_hole = 1;
    if (*srt_num == 0) {
        /* this process has nothing to receive and hence no hole */
        check_hole = 0;
        hole = 0;
    } else if (fd->hints->ds_write == ADIOI_HINT_AUTO) {
        if (*srt_num > fd->hints->ds_wr_lb) {
            /* Number of offset-length pairs is too large, making merge sort
             * expensive. Skip the sorting in hole checking and proceed with
             * read-modify-write.
             */
            check_hole = 0;
            hole = 1;
        }
        /* else: merge sort is less expensive, proceed to check_hole */
    }
    else if (fd->hints->ds_write == ADIOI_HINT_ENABLE) {
        check_hole = 0;
        hole = 1;
    }
    /* else: fd->hints->ds_write == ADIOI_HINT_DISABLE,
     * proceed to check_hole, as we must construct srt_off and srt_len.
     */

    if (check_hole) {
        /* merge the offset-length pairs of all others_req[] (already sorted
         * individually) into a single list of offset-length pairs (srt_off and
         * srt_len) in an increasing order of file offsets using a heap-merge
         * sorting algorithm.
         */
        if (*srt_off == NULL || *srt_num > malloc_srt_num) {
            /* Try to avoid malloc each round. If *srt_num is less than
             * previous round, the already allocated space can be reused.
             */
            if (*srt_off != NULL) {
                ADIOI_Free(*srt_off);
                ADIOI_Free(*srt_len);
            }
            *srt_off = (ADIO_Offset *) ADIOI_Malloc(*srt_num * sizeof(ADIO_Offset));
            *srt_len = (MPI_Count *) ADIOI_Malloc(*srt_num * sizeof(MPI_Count));
            malloc_srt_num = *srt_num;
        }

        heap_merge(others_req, recv_count, *srt_off, *srt_len, start_pos,
                   nprocs, nprocs_recv, srt_num);

        /* (*srt_num) has been updated in heap_merge() such that (*srt_off) and
         * (*srt_len) were coalesced
         */
        hole = (*srt_num > 1);
    }

    /* data sieving */
    if (fd->hints->ds_write != ADIOI_HINT_DISABLE && hole) {
        ADIO_ReadContig(fd, write_buf, range_size, MPI_BYTE, ADIO_EXPLICIT_OFFSET,
                        range_off, &status, &err);
        if (err != MPI_SUCCESS) {
            *error_code = MPIO_Err_create_code(err,
                                               MPIR_ERR_RECOVERABLE,
                                               myname, __LINE__, MPI_ERR_IO, "**ioRMWrdwr", 0);
            return;
        }

        /* Once read, holes have been filled and thus the number of
         * offset-length pairs, *srt_num, becomes one.
         */
        *srt_num = 1;
        if (*srt_off == NULL) {
            *srt_off = (ADIO_Offset *) ADIOI_Malloc(sizeof(ADIO_Offset));
            *srt_len = (MPI_Count *) ADIOI_Malloc(sizeof(MPI_Count));
            malloc_srt_num = 1;
        }
        (*srt_off)[0] = range_off;
        (*srt_len)[0] = range_size;
    }

    /* It is possible sum_recv (sum of message sizes to be received) is larger
     * than the size of collective buffer, write_buf, if writes from multiple
     * remote processes overlap. Receiving messages into overlapped regions of
     * the same write_buffer may cause a problem. To avoid it, we allocate a
     * temporary buffer big enough to receive all messages into disjointed
     * regions. Earlier in ADIOI_LUSTRE_Exch_and_write(), write_buf is already
     * allocated with twice amount of the file stripe size, with the second half
     * to be used to receive messages. If sum_recv is smalled than file stripe
     * size, we can reuse that space. But if sum_recv is bigger (an overlap
     * case, which is rare), we allocate a separate buffer of size sum_recv.
     */
    sum_recv -= recv_size[myrank];
    if (sum_recv > fd->hints->striping_unit)
        contig_buf = (char *) ADIOI_Malloc(sum_recv);
    else
        contig_buf = write_buf + fd->hints->striping_unit;

#define MEMCPY_UNPACK(x, inbuf) {                                   \
    int _k;                                                         \
    char *_ptr = (inbuf);                                           \
    MPI_Count   *mem_ptrs = others_req[x].mem_ptrs + start_pos[x];  \
    ADIO_Offset *mem_lens = others_req[x].lens     + start_pos[x];  \
    for (_k=0; _k<recv_count[x]; _k++) {                            \
        memcpy(write_buf + mem_ptrs[_k], _ptr, mem_lens[_k]);       \
        _ptr += mem_lens[_k];                                       \
    }                                                               \
}

    /* nreqs is the number of Issend and Irecv to be posted */
    nreqs = 0;
    if (fd->atomicity) {
        /* atomic mode calls blocking receive only */
        requests = (MPI_Request *) ADIOI_Malloc((nprocs_send + 1) * sizeof(MPI_Request));
    } else {
        requests =
            (MPI_Request *) ADIOI_Malloc((nprocs_send + nprocs_recv + 1) * sizeof(MPI_Request));

        /* post receives */
        /* receive messages into contig_buf, a temporary buffer */
        buf_ptr = contig_buf;
        for (i = 0; i < nprocs; i++) {
            if (recv_size[i] == 0)
                continue;
            if (i != myrank) {
                if (recv_count[i] > 1) {
                    MPI_Irecv(buf_ptr, recv_size[i], MPI_BYTE, i, ADIOI_COLL_TAG(i, iter), fd->comm,
                              &requests[nreqs++]);
                    buf_ptr += recv_size[i];
                } else {
                    /* recv_count[i] is the number of noncontiguous
                     * offset-length pairs describing the write requests of
                     * process i that fall into this aggregator's file domain.
                     * When recv_count[i] is 1, there is only one such pair,
                     * meaning the receive message is to be stored
                     * contiguously. Such message can be received directly into
                     * write_buf.
                     */
                    MPI_Irecv(write_buf + others_req[i].mem_ptrs[start_pos[i]], recv_size[i],
                              MPI_BYTE, i, ADIOI_COLL_TAG(i, iter), fd->comm, &requests[nreqs++]);
                }
            } else if (flat_buf->count == 1) {
                /* send/recv to/from self uses memcpy() */
                MEMCPY_UNPACK(i, (char *) buf + buf_idx[i]);
            }
        }
    }

    /* Post sends: if buftype is contiguous, data can be directly sent from the
     * user bufffer at location given by buf_idx. Otherwise, copy write data to
     * send_buf first and use send_buf to send.
     */
    if (flat_buf->count == 1) {
        for (i = 0; i < nprocs; i++)
            if (send_size[i] && i != myrank) {
                MPI_Issend_c((char *) buf + buf_idx[i], send_size[i],
                             MPI_BYTE, i, ADIOI_COLL_TAG(i, iter), fd->comm, &requests[nreqs++]);
            }
    } else if (nprocs_send) {
        /* If buftype is not contiguous, pack data into send_buf[], including
         * ones sent to self.
         */
        size_t send_total_size = 0;
        for (i = 0; i < nprocs; i++)
            send_total_size += send_size[i];
        send_buf = (char **) ADIOI_Malloc(nprocs * sizeof(char *));
        send_buf[0] = (char *) ADIOI_Malloc(send_total_size);
        for (i = 1; i < nprocs; i++)
            send_buf[i] = send_buf[i - 1] + send_size[i - 1];

        ADIOI_LUSTRE_Fill_send_buffer(fd, buf, flat_buf, send_buf, offset_list,
                                      len_list, send_size, requests + nreqs,
                                      sent_to_proc, nprocs, myrank,
                                      contig_access_count,
                                      send_buf_idx, curr_to_proc, done_to_proc,
                                      iter, buftype_extent);
        /* MPI_Issend calls have been posted in ADIOI_Fill_send_buffer */
        nreqs += (send_size[myrank]) ? nprocs_send - 1 : nprocs_send;
    }

    if (fd->atomicity) {
        /* In atomic mode, we must receive write data in the increasing order
         * of MPI process rank IDs,
         */
        for (i = 0; i < nprocs; i++) {
            if (recv_size[i] == 0)
                continue;
            if (i != myrank) {
                MPI_Recv(contig_buf, recv_size[i], MPI_BYTE, i, ADIOI_COLL_TAG(i, iter), fd->comm, &status);
                MEMCPY_UNPACK(i, contig_buf);
            } else {
                /* send/recv to/from self uses memcpy() */
                char *ptr = (flat_buf->count == 1) ? (char *) buf + buf_idx[i] : send_buf[i];
                MEMCPY_UNPACK(i, ptr);
            }
        }
    } else if (flat_buf->count != 1 && send_size[myrank]) {
        /* contents of user buf has been copied into send_buf[myrank] */
        MEMCPY_UNPACK(myrank, send_buf[myrank]);
    }

    if (nreqs > 0) {
#ifdef MPI_STATUSES_IGNORE
        MPI_Waitall(nreqs, requests, MPI_STATUSES_IGNORE);
#else
        MPI_Status *statuses = (MPI_Status *) ADIOI_Malloc(nreqs * sizeof(MPI_Status));
        MPI_Waitall(nreqs, requests, statuses);
        ADIOI_Free(statuses);
#endif
    }
    ADIOI_Free(requests);
    if (send_buf != NULL) {
        ADIOI_Free(send_buf[0]);
        ADIOI_Free(send_buf);
    }

    if (!fd->atomicity) {
        /* unpack from contig_buf to write_buf */
        buf_ptr = contig_buf;
        for (i = 0; i < nprocs; i++) {
            if (recv_count[i] > 1 && i != myrank) {
                /* When recv_count[i] == 1, this case has been taken care of
                 * earlier by receiving the message directly into write_buf.
                 */
                MEMCPY_UNPACK(i, buf_ptr);
                buf_ptr += recv_size[i];
            }
        }
    }

    /* free temporary receive buffer */
    if (sum_recv > fd->hints->striping_unit)
        ADIOI_Free(contig_buf);
}

#define ADIOI_BUF_INCR \
{ \
    while (buf_incr) { \
        size_in_buf = MPL_MIN(buf_incr, flat_buf_sz); \
        user_buf_idx += size_in_buf; \
        flat_buf_sz -= size_in_buf; \
        if (!flat_buf_sz) { \
            if (flat_buf_idx < (flat_buf->count - 1)) flat_buf_idx++; \
            else { \
                flat_buf_idx = 0; \
                n_buftypes++; \
            } \
            user_buf_idx = flat_buf->indices[flat_buf_idx] + \
                (ADIO_Offset)n_buftypes*(ADIO_Offset)buftype_extent;  \
            flat_buf_sz = flat_buf->blocklens[flat_buf_idx]; \
        } \
        buf_incr -= size_in_buf; \
    } \
}


#define ADIOI_BUF_COPY \
{ \
    while (size) { \
        size_in_buf = MPL_MIN(size, flat_buf_sz); \
        memcpy(&(send_buf[p][send_buf_idx[p]]), \
               ((char *) buf) + user_buf_idx, size_in_buf); \
        send_buf_idx[p] += size_in_buf; \
        user_buf_idx += size_in_buf; \
        flat_buf_sz -= size_in_buf; \
        if (!flat_buf_sz) { \
            if (flat_buf_idx < (flat_buf->count - 1)) flat_buf_idx++; \
            else { \
                flat_buf_idx = 0; \
                n_buftypes++; \
            } \
            user_buf_idx = flat_buf->indices[flat_buf_idx] + \
                (ADIO_Offset)n_buftypes*(ADIO_Offset)buftype_extent;    \
            flat_buf_sz = flat_buf->blocklens[flat_buf_idx]; \
        } \
        size -= size_in_buf; \
        buf_incr -= size_in_buf; \
    } \
    ADIOI_BUF_INCR \
}

static void ADIOI_LUSTRE_Fill_send_buffer(ADIO_File fd, const void *buf,
                                          ADIOI_Flatlist_node *flat_buf,
                                          char **send_buf,
                                          ADIO_Offset *offset_list,
                                          ADIO_Offset *len_list, MPI_Count *send_size,
                                          MPI_Request *requests,
                                          MPI_Count *sent_to_proc, int nprocs,
                                          int myrank,
                                          MPI_Count contig_access_count,
                                          ADIO_Offset *send_buf_idx,
                                          MPI_Count *curr_to_proc,
                                          MPI_Count *done_to_proc, int iter, MPI_Aint buftype_extent)
{
    /* this function is only called if buftype is not contig */
    int p;
    MPI_Count size, flat_buf_idx, buf_incr;
    MPI_Aint flat_buf_sz, size_in_buf;
    int jj, n_buftypes;
    ADIO_Offset off, len, rem_len, user_buf_idx;

    /* curr_to_proc[p] = amount of data sent to proc. p that has already
     * been accounted for so far
     * done_to_proc[p] = amount of data already sent to proc. p in
     * previous iterations
     * user_buf_idx = current location in user buffer
     * send_buf_idx[p] = current location in send_buf of proc. p
     */

    for (int i = 0; i < nprocs; i++) {
        send_buf_idx[i] = curr_to_proc[i] = 0;
        done_to_proc[i] = sent_to_proc[i];
    }
    jj = 0;

    user_buf_idx = flat_buf->indices[0];
    flat_buf_idx = 0;
    n_buftypes = 0;
    flat_buf_sz = flat_buf->blocklens[0];

    /* flat_buf_idx = current index into flattened buftype
     * flat_buf_sz = size of current contiguous component in flattened buf
     */
    for (MPI_Count i = 0; i < contig_access_count; i++) {
        off = offset_list[i];
        rem_len = (ADIO_Offset) len_list[i];

        /*this request may span to more than one process */
        while (rem_len != 0) {
            len = rem_len;
            /* NOTE: len value is modified by ADIOI_Calc_aggregator() to be no
             * longer than the single region that processor "p" is responsible
             * for.
             */
            p = ADIOI_LUSTRE_Calc_aggregator(fd, off, &len);

            if (send_buf_idx[p] < send_size[p]) {
                if (curr_to_proc[p] + len > done_to_proc[p]) {
                    if (done_to_proc[p] > curr_to_proc[p]) {
                        size = MPL_MIN(curr_to_proc[p] + len -
                                       done_to_proc[p], send_size[p] - send_buf_idx[p]);
                        buf_incr = done_to_proc[p] - curr_to_proc[p];
                        ADIOI_BUF_INCR buf_incr = (curr_to_proc[p] + len - done_to_proc[p]);
                        curr_to_proc[p] = done_to_proc[p] + size;
                    ADIOI_BUF_COPY} else {
                        size = MPL_MIN(len, send_size[p] - send_buf_idx[p]);
                        buf_incr = len;
                        curr_to_proc[p] += size;
                    ADIOI_BUF_COPY}
                    if (send_buf_idx[p] == send_size[p] && p != myrank) {
                        MPI_Issend_c(send_buf[p], send_size[p], MPI_BYTE, p,
                                     ADIOI_COLL_TAG(p, iter), fd->comm, &requests[jj++]);
                    }
                } else {
                    curr_to_proc[p] += len;
                    buf_incr = len;
                ADIOI_BUF_INCR}
            } else {
                buf_incr = len;
            ADIOI_BUF_INCR}
            off += len;
            rem_len -= len;
        }
    }
    for (int i = 0; i < nprocs; i++)
        if (send_size[i])
            sent_to_proc[i] = curr_to_proc[i];
}

/* This function calls ADIOI_OneSidedWriteAggregation iteratively to
 * essentially pack stripes of data into the collective buffer and then
 * flush the collective buffer to the file when fully packed, repeating this
 * process until all the data is written to the file.
 */
static void ADIOI_LUSTRE_IterateOneSided(ADIO_File fd, const void *buf,
                                         ADIO_Offset * offset_list, ADIO_Offset * len_list,
                                         MPI_Count contig_access_count,
                                         MPI_Count currentValidDataIndex, MPI_Aint count,
                                         int file_ptr_type, ADIO_Offset offset,
                                         ADIO_Offset start_offset, ADIO_Offset end_offset,
                                         ADIO_Offset firstFileOffset, ADIO_Offset lastFileOffset,
                                         MPI_Datatype buftype, int myrank, int *error_code)
{
    int i;
    int striping_unit = fd->hints->striping_unit;
    int stripesPerAgg = fd->hints->cb_buffer_size / striping_unit;
    if (stripesPerAgg == 0) {
        /* The striping unit is larger than the collective buffer size
         * therefore we must abort since the buffer has already been
         * allocated during the open.
         */
        FPRINTF(stderr, "Error: The collective buffer size %d is less "
                "than the striping unit size %d - the ROMIO "
                "Lustre one-sided write aggregation algorithm "
                "cannot continue.\n", fd->hints->cb_buffer_size, striping_unit);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Based on the co_ratio the number of aggregators we can use is the number of
     * stripes used in the file times this co_ratio - each stripe is written by
     * co_ratio aggregators this information is contained in the striping_info.
     */
    int numStripedAggs = fd->hints->cb_nodes;

    int orig_cb_nodes = fd->hints->cb_nodes;
    fd->hints->cb_nodes = numStripedAggs;

    /* Declare ADIOI_OneSidedStripeParms here - these parameters will be locally managed
     * for this invocation of ADIOI_LUSTRE_IterateOneSided.  This will allow for concurrent
     * one-sided collective writes via multi-threading as well as multiple communicators.
     */
    ADIOI_OneSidedStripeParms stripeParms;
    stripeParms.stripeSize = striping_unit;
    stripeParms.stripedLastFileOffset = lastFileOffset;
    stripeParms.iWasUsedStripingAgg = 0;
    stripeParms.numStripesUsed = 0;
    stripeParms.amountOfStripedDataExpected = 0;
    stripeParms.bufTypeExtent = 0;
    stripeParms.lastDataTypeExtent = 0;
    stripeParms.lastFlatBufIndice = 0;
    stripeParms.lastIndiceOffset = 0;

    /* The general algorithm here is to divide the file up into segments, a segment
     * being defined as a contiguous region of the file which has up to one occurrence
     * of each stripe - the data for each stripe being written out by a particular
     * aggregator.  The segmentLen is the maximum size in bytes of each segment
     * (stripeSize*number of aggs).  Iteratively call ADIOI_OneSidedWriteAggregation
     * for each segment to aggregate the data to the collective buffers, but only do
     * the actual write (via flushCB stripe parm) once stripesPerAgg stripes
     * have been packed or the aggregation for all the data is complete, minimizing
     * synchronization.
     */
    stripeParms.segmentLen = ((ADIO_Offset) numStripedAggs) * ((ADIO_Offset) (striping_unit));

    /* These arrays define the file offsets for the stripes for a given segment - similar
     * to the concept of file domains in GPFS, essentially file domains for the segment.
     */
    ADIO_Offset *segment_stripe_start =
        (ADIO_Offset *) ADIOI_Malloc(numStripedAggs * sizeof(ADIO_Offset));
    ADIO_Offset *segment_stripe_end =
        (ADIO_Offset *) ADIOI_Malloc(numStripedAggs * sizeof(ADIO_Offset));

    /* Find the actual range of stripes in the file that have data in the offset
     * ranges being written -- skip holes at the front and back of the file.
     */
    MPI_Count currentOffsetListIndex = 0;
    MPI_Count fileSegmentIter = 0;
    MPI_Count startingStripeWithData = 0;
    MPI_Count foundStartingStripeWithData = 0;
    while (!foundStartingStripeWithData) {
        if (((startingStripeWithData + 1) * (ADIO_Offset) (striping_unit)) > firstFileOffset)
            foundStartingStripeWithData = 1;
        else
            startingStripeWithData++;
    }

    ADIO_Offset currentSegementOffset =
        (ADIO_Offset) startingStripeWithData * (ADIO_Offset) (striping_unit);

    MPI_Count numSegments =
        ((lastFileOffset + (ADIO_Offset) 1 - currentSegementOffset) / stripeParms.segmentLen);
    if ((lastFileOffset + (ADIO_Offset) 1 - currentSegementOffset) % stripeParms.segmentLen > 0)
        numSegments++;

    /* To support read-modify-write use a while-loop to redo the aggregation if necessary
     * to fill in the holes.
     */
    int doAggregation = 1;
    int holeFound = 0;

    /* Remember romio_onesided_no_rmw setting if we have to re-do
     * the aggregation if holes are found.
     */
    int prev_romio_onesided_no_rmw = fd->romio_onesided_no_rmw;

    while (doAggregation) {

        int totalDataWrittenLastRound = 0;

        /* This variable tracks how many segment stripes we have packed into the agg
         * buffers so we know when to flush to the file system.
         */
        stripeParms.segmentIter = 0;

        /* stripeParms.stripesPerAgg is the number of stripes to aggregate before doing a flush.
         */
        stripeParms.stripesPerAgg = stripesPerAgg;
        if (stripeParms.stripesPerAgg > numSegments)
            stripeParms.stripesPerAgg = numSegments;

        for (fileSegmentIter = 0; fileSegmentIter < numSegments; fileSegmentIter++) {

            MPI_Count dataWrittenThisRound = 0;

            /* Define the segment range in terms of file offsets.
             */
            ADIO_Offset segmentFirstFileOffset = currentSegementOffset;
            if ((currentSegementOffset + stripeParms.segmentLen - (ADIO_Offset) 1) > lastFileOffset)
                currentSegementOffset = lastFileOffset;
            else
                currentSegementOffset += (stripeParms.segmentLen - (ADIO_Offset) 1);
            ADIO_Offset segmentLastFileOffset = currentSegementOffset;
            currentSegementOffset++;

            ADIO_Offset segment_stripe_offset = segmentFirstFileOffset;
            for (i = 0; i < numStripedAggs; i++) {
                if (firstFileOffset > segment_stripe_offset)
                    segment_stripe_start[i] = firstFileOffset;
                else
                    segment_stripe_start[i] = segment_stripe_offset;
                if ((segment_stripe_offset + (ADIO_Offset) (striping_unit)) > lastFileOffset)
                    segment_stripe_end[i] = lastFileOffset;
                else
                    segment_stripe_end[i] =
                        segment_stripe_offset + (ADIO_Offset) (striping_unit) - (ADIO_Offset) 1;
                segment_stripe_offset += (ADIO_Offset) (striping_unit);
            }

            /* In the interest of performance for non-contiguous data with large offset lists
             * essentially modify the given offset and length list appropriately for this segment
             * and then pass pointers to the sections of the lists being used for this segment
             * to ADIOI_OneSidedWriteAggregation.  Remember how we have modified the list for this
             * segment, and then restore it appropriately after processing for this segment has
             * concluded, so it is ready for the next segment.
             */
            MPI_Count segmentContigAccessCount = 0;
            MPI_Count startingOffsetListIndex = -1;
            MPI_Count endingOffsetListIndex = -1;
            ADIO_Offset startingOffsetAdvancement = 0;
            ADIO_Offset startingLenTrim = 0;
            ADIO_Offset endingLenTrim = 0;

            while (((offset_list[currentOffsetListIndex] +
                     ((ADIO_Offset) (len_list[currentOffsetListIndex])) - (ADIO_Offset) 1) <
                    segmentFirstFileOffset) && (currentOffsetListIndex < (contig_access_count - 1)))
                currentOffsetListIndex++;
            startingOffsetListIndex = currentOffsetListIndex;
            endingOffsetListIndex = currentOffsetListIndex;
            MPI_Count offsetInSegment = 0;
            ADIO_Offset offsetStart = offset_list[currentOffsetListIndex];
            ADIO_Offset offsetEnd =
                (offset_list[currentOffsetListIndex] +
                 ((ADIO_Offset) (len_list[currentOffsetListIndex])) - (ADIO_Offset) 1);

            if (len_list[currentOffsetListIndex] == 0)
                offsetInSegment = 0;
            else if ((offsetStart >= segmentFirstFileOffset) &&
                     (offsetStart <= segmentLastFileOffset)) {
                offsetInSegment = 1;
            } else if ((offsetEnd >= segmentFirstFileOffset) &&
                       (offsetEnd <= segmentLastFileOffset)) {
                offsetInSegment = 1;
            } else if ((offsetStart <= segmentFirstFileOffset) &&
                       (offsetEnd >= segmentLastFileOffset)) {
                offsetInSegment = 1;
            }

            if (!offsetInSegment) {
                segmentContigAccessCount = 0;
            } else {
                /* We are in the segment, advance currentOffsetListIndex until we are out of segment.
                 */
                segmentContigAccessCount = 1;

                while ((offset_list[currentOffsetListIndex] <= segmentLastFileOffset) &&
                       (currentOffsetListIndex < contig_access_count)) {
                    dataWrittenThisRound += len_list[currentOffsetListIndex];
                    currentOffsetListIndex++;
                }

                if (currentOffsetListIndex > startingOffsetListIndex) {
                    /* If we did advance, if we are at the end need to check if we are still in segment.
                     */
                    if (currentOffsetListIndex == contig_access_count) {
                        currentOffsetListIndex--;
                    } else if (offset_list[currentOffsetListIndex] > segmentLastFileOffset) {
                        /* We advanced into the last one and it still in the segment.
                         */
                        currentOffsetListIndex--;
                    } else {
                        dataWrittenThisRound += len_list[currentOffsetListIndex];
                    }
                    segmentContigAccessCount += (currentOffsetListIndex - startingOffsetListIndex);
                    endingOffsetListIndex = currentOffsetListIndex;
                }
            }

            if (segmentContigAccessCount > 0) {
                /* Trim edges here so all data in the offset list range fits exactly in the segment.
                 */
                if (offset_list[startingOffsetListIndex] < segmentFirstFileOffset) {
                    startingOffsetAdvancement =
                        segmentFirstFileOffset - offset_list[startingOffsetListIndex];
                    offset_list[startingOffsetListIndex] += startingOffsetAdvancement;
                    dataWrittenThisRound -= startingOffsetAdvancement;
                    startingLenTrim = startingOffsetAdvancement;
                    len_list[startingOffsetListIndex] -= startingLenTrim;
                }

                if ((offset_list[endingOffsetListIndex] +
                     ((ADIO_Offset) (len_list[endingOffsetListIndex])) - (ADIO_Offset) 1) >
                    segmentLastFileOffset) {
                    endingLenTrim =
                        offset_list[endingOffsetListIndex] +
                        ((ADIO_Offset) (len_list[endingOffsetListIndex])) - (ADIO_Offset) 1 -
                        segmentLastFileOffset;
                    len_list[endingOffsetListIndex] -= endingLenTrim;
                    dataWrittenThisRound -= endingLenTrim;
                }
            }

            int holeFoundThisRound = 0;

            /* Once we have packed the collective buffers do the actual write.
             */
            if ((stripeParms.segmentIter == (stripeParms.stripesPerAgg - 1)) ||
                (fileSegmentIter == (numSegments - 1))) {
                stripeParms.flushCB = 1;
            } else
                stripeParms.flushCB = 0;

            stripeParms.firstStripedWriteCall = 0;
            stripeParms.lastStripedWriteCall = 0;
            if (fileSegmentIter == 0) {
                stripeParms.firstStripedWriteCall = 1;
            } else if (fileSegmentIter == (numSegments - 1))
                stripeParms.lastStripedWriteCall = 1;

            /* The difference in calls to ADIOI_OneSidedWriteAggregation is based on the whether the buftype is
             * contiguous.  The algorithm tracks the position in the source buffer when called
             * multiple times --  in the case of contiguous data this is simple and can be externalized with
             * a buffer offset, in the case of non-contiguous data this is complex and the state must be tracked
             * internally, therefore no external buffer offset.  Care was taken to minimize
             * ADIOI_OneSidedWriteAggregation changes at the expense of some added complexity to the caller.
             */
            ADIOI_Flatlist_node *flat_buf = ADIOI_Flatten_and_find(buftype);
            if (flat_buf->count == 1) { /* buftype is contiguous */
                ADIOI_OneSidedWriteAggregation(fd,
                                               &(offset_list[startingOffsetListIndex]),
                                               &(len_list[startingOffsetListIndex]),
                                               segmentContigAccessCount,
                                               buf + totalDataWrittenLastRound, buftype,
                                               error_code, segmentFirstFileOffset,
                                               segmentLastFileOffset, currentValidDataIndex,
                                               segment_stripe_start, segment_stripe_end,
                                               &holeFoundThisRound, &stripeParms);
                /* numNonZeroDataOffsets is not used in ADIOI_OneSidedWriteAggregation()? */
            } else {
                ADIOI_OneSidedWriteAggregation(fd,
                                               &(offset_list[startingOffsetListIndex]),
                                               &(len_list[startingOffsetListIndex]),
                                               segmentContigAccessCount, buf, buftype, error_code,
                                               segmentFirstFileOffset, segmentLastFileOffset,
                                               currentValidDataIndex, segment_stripe_start,
                                               segment_stripe_end, &holeFoundThisRound,
                                               &stripeParms);
            }

            if (stripeParms.flushCB) {
                stripeParms.segmentIter = 0;
                if (stripesPerAgg > (numSegments - fileSegmentIter - 1))
                    stripeParms.stripesPerAgg = numSegments - fileSegmentIter - 1;
                else
                    stripeParms.stripesPerAgg = stripesPerAgg;
            } else
                stripeParms.segmentIter++;

            if (holeFoundThisRound)
                holeFound = 1;

            /* If we know we won't be doing a pre-read in a subsequent call to
             * ADIOI_OneSidedWriteAggregation which will have a barrier to keep
             * feeder ranks from doing rma to the collective buffer before the
             * write completes that we told it do with the stripeParms.flushCB
             * flag then we need to do a barrier here.
             */
            if (!fd->romio_onesided_always_rmw && stripeParms.flushCB) {
                if (fileSegmentIter < (numSegments - 1)) {
                    MPI_Barrier(fd->comm);
                }
            }

            /* Restore the offset_list and len_list to values that are ready for the
             * next iteration.
             */
            if (segmentContigAccessCount > 0) {
                offset_list[endingOffsetListIndex] += len_list[endingOffsetListIndex];
                len_list[endingOffsetListIndex] = endingLenTrim;
            }
            totalDataWrittenLastRound += dataWrittenThisRound;
        }       // fileSegmentIter for-loop

        /* Check for holes in the data unless romio_onesided_no_rmw is set.
         * If a hole is found redo the entire aggregation and write.
         */
        if (!fd->romio_onesided_no_rmw) {
            int anyHolesFound = 0;
            MPI_Allreduce(&holeFound, &anyHolesFound, 1, MPI_INT, MPI_MAX, fd->comm);

            if (anyHolesFound) {
                ADIOI_Free(offset_list);
                ADIOI_Calc_my_off_len(fd, count, buftype, file_ptr_type, offset,
                                      &offset_list, &len_list, &start_offset,
                                      &end_offset, &contig_access_count);

                currentSegementOffset =
                    (ADIO_Offset) startingStripeWithData *(ADIO_Offset) (striping_unit);
                fd->romio_onesided_always_rmw = 1;
                fd->romio_onesided_no_rmw = 1;

                /* Holes are found in the data and the user has not set
                 * romio_onesided_no_rmw --- set romio_onesided_always_rmw to 1
                 * and redo the entire aggregation and write and if the user has
                 * romio_onesided_inform_rmw set then inform him of this condition
                 * and behavior.
                 */
                if (fd->romio_onesided_inform_rmw && (myrank == 0)) {
                    FPRINTF(stderr, "Information: Holes found during one-sided "
                            "write aggregation algorithm --- re-running one-sided "
                            "write aggregation with ROMIO_ONESIDED_ALWAYS_RMW set to 1.\n");
                }
            } else
                doAggregation = 0;
        } else
            doAggregation = 0;
    }   // while doAggregation
    fd->romio_onesided_no_rmw = prev_romio_onesided_no_rmw;

    ADIOI_Free(segment_stripe_start);
    ADIOI_Free(segment_stripe_end);

    fd->hints->cb_nodes = orig_cb_nodes;

}
