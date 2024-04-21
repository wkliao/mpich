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
if (fd->hints->fs_hints.lustre.lock_ahead_write) {                           \
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
                                        int contig_access_count,
                                        int *striping_info,
                                        ADIO_Offset **buf_idx, int *error_code);
static void ADIOI_LUSTRE_Fill_send_buffer(ADIO_File fd, const void *buf,
                                          ADIOI_Flatlist_node *flat_buf,
                                          char **send_buf,
                                          ADIO_Offset *offset_list,
                                          ADIO_Offset *len_list, int *send_size,
                                          MPI_Request *requests,
                                          int *sent_to_proc, int nprocs,
                                          int myrank, int contig_access_count,
                                          int *striping_info,
                                          ADIO_Offset *send_buf_idx,
                                          int *curr_to_proc,
                                          int *done_to_proc, int iter, MPI_Aint buftype_extent);
static void ADIOI_LUSTRE_W_Exchange_data(ADIO_File fd, const void *buf,
                                         char *write_buf,
                                         ADIOI_Flatlist_node *flat_buf,
                                         ADIO_Offset *offset_list,
                                         ADIO_Offset *len_list, int *send_size,
                                         int *recv_size, ADIO_Offset off,
                                         int size, int *count,
                                         int *start_pos,
                                         int *sent_to_proc, int nprocs,
                                         int myrank,
                                         int contig_access_count,
                                         int *striping_info,
                                         ADIOI_Access *others_req,
                                         ADIO_Offset *send_buf_idx,
                                         int *curr_to_proc,
                                         int *done_to_proc,
                                         int iter, MPI_Aint buftype_extent,
                                         ADIO_Offset *buf_idx,
                                         ADIO_Offset **srt_off, int **srt_len, int *srt_num,
                                         int *error_code);
void ADIOI_Heap_merge(ADIOI_Access *others_req, int *count,
                      ADIO_Offset *srt_off, int *srt_len, int *start_pos,
                      int nprocs, int nprocs_recv, int total_elements);

static void ADIOI_LUSTRE_IterateOneSided(ADIO_File fd, const void *buf, int *striping_info,
                                         ADIO_Offset * offset_list, ADIO_Offset * len_list,
                                         int contig_access_count, int currentValidDataIndex,
                                         int count, int file_ptr_type, ADIO_Offset offset,
                                         ADIO_Offset start_offset, ADIO_Offset end_offset,
                                         ADIO_Offset firstFileOffset, ADIO_Offset lastFileOffset,
                                         MPI_Datatype buftype, int myrank, int *error_code);

void ADIOI_LUSTRE_WriteStridedColl(ADIO_File fd, const void *buf, MPI_Aint count,
                                   MPI_Datatype buftype,
                                   int file_ptr_type, ADIO_Offset offset,
                                   ADIO_Status *status, int *error_code)
{
    /* Uses a generalized version of the extended two-phase method described
     * in "An Extended Two-Phase Method for Accessing Sections of
     * Out-of-Core Arrays", Rajeev Thakur and Alok Choudhary,
     * Scientific Programming, (5)4:301--317, Winter 1996.
     * http://www.mcs.anl.gov/home/thakur/ext2ph.ps
     */

    ADIOI_Access *my_req;
    /* array of nprocs access structures, one for each other process has
     * this process's request */

    ADIOI_Access *others_req;
    /* array of nprocs access structures, one for each other process
     * whose request is written by this process. */

    int i, nprocs, myrank, do_collect = 0;
    int contig_access_count = 0, buftype_is_contig, interleave_count = 0;
    int *count_my_req_per_proc, count_my_req_procs;
    int *count_others_req_per_proc, count_others_req_procs;
    ADIO_Offset orig_fp, start_offset, end_offset, off;
    ADIO_Offset *offset_list = NULL, *st_offsets = NULL, *end_offsets = NULL;
    ADIO_Offset *len_list = NULL;
    int striping_info[3];
    ADIO_Offset **buf_idx = NULL;
    int old_error, tmp_error;
    ADIO_Offset *lustre_offsets0, *lustre_offsets, *count_sizes = NULL;
    ADIOI_Flatlist_node *flat_buf;

    MPI_Comm_size(fd->comm, &nprocs);
    MPI_Comm_rank(fd->comm, &myrank);

    orig_fp = fd->fp_ind;

    /* IO pattern identification if cb_write isn't disabled */
    if (fd->hints->cb_write != ADIOI_HINT_DISABLE) {
        /* For this process's request, calculate the list of offsets and
         * lengths in the file and determine the start and end offsets.
         * Note: end_offset points to the last byte-offset to be accessed.
         * e.g., if start_offset=0 and 100 bytes to be read, end_offset=99
         */
        ADIOI_Calc_my_off_len(fd, count, buftype, file_ptr_type, offset,
                              &offset_list, &len_list, &start_offset,
                              &end_offset, &contig_access_count);

        /* each process communicates its start and end offsets to other
         * processes. The result is an array each of start and end offsets
         * stored in order of process rank.
         */
        st_offsets = (ADIO_Offset *) ADIOI_Malloc(nprocs * 2 * sizeof(ADIO_Offset));
        end_offsets = st_offsets + nprocs;
        ADIO_Offset my_count_size = 0;
        /* One-sided aggregation needs the amount of data per rank as well
         * because the difference in starting and ending offsets for 1 byte is
         * 0 the same as 0 bytes so it cannot be distinguished.
         */
        if ((fd->romio_write_aggmethod == 1) || (fd->romio_write_aggmethod == 2)) {
            count_sizes = (ADIO_Offset *) ADIOI_Malloc(nprocs * sizeof(ADIO_Offset));
            MPI_Count buftype_size;
            MPI_Type_size_x(buftype, &buftype_size);
            my_count_size = (ADIO_Offset) count *(ADIO_Offset) buftype_size;
        }
        if (fd->romio_tunegather) {
            if ((fd->romio_write_aggmethod == 1) || (fd->romio_write_aggmethod == 2)) {
                lustre_offsets0 = (ADIO_Offset *) ADIOI_Malloc(6 * nprocs * sizeof(ADIO_Offset));
                lustre_offsets = lustre_offsets0 + 3 * nprocs;
                for (i = 0; i < nprocs; i++) {
                    lustre_offsets0[i * 3] = 0;
                    lustre_offsets0[i * 3 + 1] = 0;
                    lustre_offsets0[i * 3 + 2] = 0;
                }
                lustre_offsets0[myrank * 3] = start_offset;
                lustre_offsets0[myrank * 3 + 1] = end_offset;
                lustre_offsets0[myrank * 3 + 2] = my_count_size;
                MPI_Allreduce(lustre_offsets0, lustre_offsets, nprocs * 3, ADIO_OFFSET, MPI_MAX,
                              fd->comm);
                for (i = 0; i < nprocs; i++) {
                    st_offsets[i] = lustre_offsets[i * 3];
                    end_offsets[i] = lustre_offsets[i * 3 + 1];
                    count_sizes[i] = lustre_offsets[i * 3 + 2];
                }
            } else {
                lustre_offsets0 = (ADIO_Offset *) ADIOI_Malloc(4 * nprocs * sizeof(ADIO_Offset));
                lustre_offsets = lustre_offsets0 + 2 * nprocs;
                for (i = 0; i < nprocs; i++) {
                    lustre_offsets0[i * 2] = 0;
                    lustre_offsets0[i * 2 + 1] = 0;
                }
                lustre_offsets0[myrank * 2] = start_offset;
                lustre_offsets0[myrank * 2 + 1] = end_offset;

                MPI_Allreduce(lustre_offsets0, lustre_offsets, nprocs * 2, ADIO_OFFSET, MPI_MAX,
                              fd->comm);

                for (i = 0; i < nprocs; i++) {
                    st_offsets[i] = lustre_offsets[i * 2];
                    end_offsets[i] = lustre_offsets[i * 2 + 1];
                }
            }
            ADIOI_Free(lustre_offsets0);
        } else {
            MPI_Allgather(&start_offset, 1, ADIO_OFFSET, st_offsets, 1, ADIO_OFFSET, fd->comm);
            MPI_Allgather(&end_offset, 1, ADIO_OFFSET, end_offsets, 1, ADIO_OFFSET, fd->comm);
            if ((fd->romio_write_aggmethod == 1) || (fd->romio_write_aggmethod == 2)) {
                MPI_Allgather(&my_count_size, 1, ADIO_OFFSET, count_sizes, 1,
                              ADIO_OFFSET, fd->comm);
            }
        }
        /* are the accesses of different processes interleaved? */
        for (i = 1; i < nprocs; i++)
            if ((st_offsets[i] < end_offsets[i - 1]) && (st_offsets[i] <= end_offsets[i]))
                interleave_count++;
        /* This is a rudimentary check for interleaving, but should suffice
         * for the moment. */

        /* Two typical access patterns can benefit from collective write.
         *   1) the processes are interleaved, and
         *   2) the req size is small.
         */
        if (interleave_count > 0) {
            do_collect = 1;
        } else {
            do_collect = ADIOI_LUSTRE_Docollect(fd, contig_access_count, len_list, nprocs);
        }
    }

    flat_buf = ADIOI_Flatten_and_find(buftype);
    if (flat_buf->count == 1) /* actually contiguous */
        buftype_is_contig = 1;
    else
        buftype_is_contig = 0;

    /* Decide if collective I/O should be done */
    if ((!do_collect && fd->hints->cb_write == ADIOI_HINT_AUTO) ||
        fd->hints->cb_write == ADIOI_HINT_DISABLE) {
        int filetype_is_contig;
        ADIOI_Flatlist_node *flat_ftype;

        /* use independent accesses */
        if (fd->hints->cb_write != ADIOI_HINT_DISABLE) {
            ADIOI_Free(offset_list);
            ADIOI_Free(st_offsets);
            if ((fd->romio_write_aggmethod == 1) || (fd->romio_write_aggmethod == 2))
                ADIOI_Free(count_sizes);
        }

        fd->fp_ind = orig_fp;
        flat_ftype = ADIOI_Flatten_and_find(fd->filetype);
        if (flat_ftype->count == 1) /* actually contiguous */
            filetype_is_contig = 1;
        else
            filetype_is_contig = 0;

        if (buftype_is_contig && filetype_is_contig) {
            if (file_ptr_type == ADIO_EXPLICIT_OFFSET) {
                off = fd->disp + (ADIO_Offset) (fd->etype_size) * offset;
                ADIO_WriteContig(fd, buf, count, buftype,
                                 ADIO_EXPLICIT_OFFSET, off, status, error_code);
            } else
                ADIO_WriteContig(fd, buf, count, buftype, ADIO_INDIVIDUAL, 0, status, error_code);
        } else {
            ADIO_WriteStrided(fd, buf, count, buftype, file_ptr_type, offset, status, error_code);
        }
        return;
    }

    ADIO_Offset lastFileOffset = 0, firstFileOffset = -1;
    int currentValidDataIndex = 0;
    if ((fd->romio_write_aggmethod == 1) || (fd->romio_write_aggmethod == 2)) {
        /* Take out the 0-data offsets by shifting the indexes with data to the front
         * and keeping track of the valid data index for use as the length.
         */
        for (i = 0; i < nprocs; i++) {
            if (count_sizes[i] > 0) {
                st_offsets[currentValidDataIndex] = st_offsets[i];
                end_offsets[currentValidDataIndex] = end_offsets[i];

                lastFileOffset = MPL_MAX(lastFileOffset, end_offsets[currentValidDataIndex]);
                if (firstFileOffset == -1)
                    firstFileOffset = st_offsets[currentValidDataIndex];
                else
                    firstFileOffset = MPL_MIN(firstFileOffset, st_offsets[currentValidDataIndex]);

                currentValidDataIndex++;
            }
        }
    }

    /* Get Lustre hints information */
    ADIOI_LUSTRE_Get_striping_info(fd, striping_info, 1);
    /* If the user has specified to use a one-sided aggregation method then do
     * that at this point instead of the two-phase I/O.
     */
    if ((fd->romio_write_aggmethod == 1) || (fd->romio_write_aggmethod == 2)) {

        ADIOI_LUSTRE_IterateOneSided(fd, buf, striping_info, offset_list, len_list,
                                     contig_access_count, currentValidDataIndex, count,
                                     file_ptr_type, offset, start_offset, end_offset,
                                     firstFileOffset, lastFileOffset, buftype, myrank, error_code);

        ADIOI_Free(offset_list);
        ADIOI_Free(st_offsets);
        ADIOI_Free(count_sizes);
        goto fn_exit;
    }   // onesided aggregation

    /* calculate what portions of the access requests of this process are
     * located in which process
     */
    ADIOI_LUSTRE_Calc_my_req(fd, offset_list, len_list, contig_access_count,
                             striping_info, nprocs, &count_my_req_procs,
                             &count_my_req_per_proc, &my_req, &buf_idx);

    /* based on everyone's my_req, calculate what requests of other processes
     * will be accessed by this process.
     * count_others_req_procs = number of processes whose requests (including
     * this process itself) will be accessed by this process
     * count_others_req_per_proc[i] indicates how many separate contiguous
     * requests of proc. i will be accessed by this process.
     */

    ADIOI_Calc_others_req(fd, count_my_req_procs, count_my_req_per_proc,
                          my_req, nprocs, myrank, &count_others_req_procs,
                          &count_others_req_per_proc, &others_req);

    /* exchange data and write in sizes of no more than stripe_size. */
    ADIOI_LUSTRE_Exch_and_write(fd, buf, buftype, flat_buf, nprocs, myrank,
                                others_req, my_req, offset_list, len_list,
                                contig_access_count, striping_info, buf_idx, error_code);

    /* If this collective write is followed by an independent write,
     * it's possible to have those subsequent writes on other processes
     * race ahead and sneak in before the read-modify-write completes.
     * We carry out a collective communication at the end here so no one
     * can start independent i/o before collective I/O completes.
     *
     * need to do some gymnastics with the error codes so that if something
     * went wrong, all processes report error, but if a process has a more
     * specific error code, we can still have that process report the
     * additional information */

    old_error = *error_code;
    if (*error_code != MPI_SUCCESS)
        *error_code = MPI_ERR_IO;

    /* optimization: if only one process performing i/o, we can perform
     * a less-expensive Bcast  */
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

    /* free all memory allocated for collective I/O */
    /* free others_req */
    ADIOI_LUSTRE_Free_my_req(nprocs, count_my_req_per_proc, my_req, buf_idx);
    ADIOI_Free_others_req(nprocs, count_others_req_per_proc, others_req);
    ADIOI_Free(offset_list);
    ADIOI_Free(st_offsets);

  fn_exit:
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

/* If successful, error_code is set to MPI_SUCCESS.  Otherwise an error
 * code is created and returned in error_code.
 */
static void ADIOI_LUSTRE_Exch_and_write(ADIO_File fd, const void *buf,
                                        MPI_Datatype buftype, ADIOI_Flatlist_node *flat_buf,
                                        int nprocs, int myrank, ADIOI_Access *others_req,
                                        ADIOI_Access *my_req,
                                        ADIO_Offset *offset_list,
                                        ADIO_Offset *len_list,
                                        int contig_access_count,
                                        int *striping_info, ADIO_Offset **buf_idx, int *error_code)
{
    /* Send data to appropriate processes and write in sizes of no more
     * than lustre stripe_size.
     * The idea is to reduce the amount of extra memory required for
     * collective I/O. If all data were written all at once, which is much
     * easier, it would require temp space more than the size of user_buf,
     * which is often unacceptable. For example, to write a distributed
     * array to a file, where each local array is 8Mbytes, requiring
     * at least another 8Mbytes of temp space is unacceptable.
     */

    int i, j, m, flag, ntimes = 1, max_ntimes;
    ADIO_Offset st_loc = -1, end_loc = -1, min_st_loc, max_end_loc;
    ADIO_Offset off, req_off, send_off, iter_st_off, *off_list;
    ADIO_Offset max_size, step_size = 0;
    int real_size, req_len, send_len;
    int *recv_curr_offlen_ptr, *recv_count, *recv_size;
    int *send_curr_offlen_ptr, *send_size;
    int *sent_to_proc, *recv_start_pos;
    int *curr_to_proc, *done_to_proc;
    ADIO_Offset *send_buf_idx, *this_buf_idx;
    char *write_buf = NULL;
    MPI_Status status;
    MPI_Aint lb, buftype_extent;
    int stripe_size = striping_info[0], avail_cb_nodes = striping_info[2];
    ADIO_Offset *srt_off = NULL;
    int *srt_len = NULL;
    int srt_num = 0;
    ADIO_Offset block_offset;
    int block_len;

    *error_code = MPI_SUCCESS;  /* changed below if error */
    /* only I/O errors are currently reported */

    /* calculate the number of writes of stripe size to be done.
     * That gives the no. of communication phases as well.
     * Note:
     *   Because we redistribute data in stripe-contiguous pattern for Lustre,
     *   each process has the same no. of communication phases.
     */

    for (i = 0; i < nprocs; i++) {
        if (others_req[i].count) {
            st_loc = others_req[i].offsets[0];
            end_loc = others_req[i].offsets[0];
            break;
        }
    }
    for (i = 0; i < nprocs; i++) {
        for (j = 0; j < others_req[i].count; j++) {
            st_loc = MPL_MIN(st_loc, others_req[i].offsets[j]);
            end_loc = MPL_MAX(end_loc, (others_req[i].offsets[j] + others_req[i].lens[j] - 1));
        }
    }
    /* this process does no writing. */
    if ((st_loc == -1) && (end_loc == -1))
        ntimes = 0;
    MPI_Allreduce(&end_loc, &max_end_loc, 1, MPI_LONG_LONG_INT, MPI_MAX, fd->comm);
    /* avoid min_st_loc be -1 */
    if (st_loc == -1)
        st_loc = max_end_loc;
    MPI_Allreduce(&st_loc, &min_st_loc, 1, MPI_LONG_LONG_INT, MPI_MIN, fd->comm);
    /* align downward */
    min_st_loc -= min_st_loc % (ADIO_Offset) stripe_size;

    /* Each time, only avail_cb_nodes number of IO clients perform IO,
     * so, step_size=avail_cb_nodes*stripe_size IO will be performed at most,
     * and ntimes=whole_file_portion/step_size
     */
    step_size = (ADIO_Offset) avail_cb_nodes *stripe_size;
    max_ntimes = (max_end_loc - min_st_loc + 1) / step_size
        + (((max_end_loc - min_st_loc + 1) % step_size) ? 1 : 0);
/*     max_ntimes = (int)((max_end_loc - min_st_loc) / step_size + 1); */
    if (ntimes)
        write_buf = (char *) ADIOI_Malloc(stripe_size);

    /* calculate the start offset for each iteration */
    off_list = (ADIO_Offset *) ADIOI_Malloc((max_ntimes + 2 * nprocs) * sizeof(ADIO_Offset));
    send_buf_idx = off_list + max_ntimes;
    this_buf_idx = send_buf_idx + nprocs;

    for (m = 0; m < max_ntimes; m++)
        off_list[m] = max_end_loc;
    for (i = 0; i < nprocs; i++) {
        for (j = 0; j < others_req[i].count; j++) {
            req_off = others_req[i].offsets[j];
            m = (int) ((req_off - min_st_loc) / step_size);
            off_list[m] = MPL_MIN(off_list[m], req_off);
        }
    }

    recv_curr_offlen_ptr = (int *) ADIOI_Calloc(nprocs * 9, sizeof(int));
    send_curr_offlen_ptr = recv_curr_offlen_ptr + nprocs;
    /* their use is explained below. calloc initializes to 0. */

    recv_count = send_curr_offlen_ptr + nprocs;
    /* to store count of how many off-len pairs per proc are satisfied
     * in an iteration. */

    send_size = recv_count + nprocs;
    /* total size of data to be sent to each proc. in an iteration.
     * Of size nprocs so that I can use MPI_Alltoall later. */

    recv_size = send_size + nprocs;
    /* total size of data to be recd. from each proc. in an iteration. */

    sent_to_proc = recv_size + nprocs;
    /* amount of data sent to each proc so far. Used in
     * ADIOI_Fill_send_buffer. initialized to 0 here. */

    curr_to_proc = sent_to_proc + nprocs;
    done_to_proc = curr_to_proc + nprocs;
    /* Above three are used in ADIOI_Fill_send_buffer */

    recv_start_pos = done_to_proc + nprocs;
    /* used to store the starting value of recv_curr_offlen_ptr[i] in
     * this iteration */

    flat_buf = ADIOI_Flatten_and_find(buftype);

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

    iter_st_off = min_st_loc;

    for (m = 0; m < max_ntimes; m++) {
        /* go through all others_req and my_req to check which will be received
         * and sent in this iteration.
         */

        /* Note that MPI guarantees that displacements in filetypes are in
         * monotonically nondecreasing order and that, for writes, the
         * filetypes cannot specify overlapping regions in the file. This
         * simplifies implementation a bit compared to reads. */

        /*
         * off         = start offset in the file for the data to be written in
         * this iteration
         * iter_st_off = start offset of this iteration
         * real_size   = size of data written (bytes) corresponding to off
         * max_size    = possible maximum size of data written in this iteration
         * req_off     = offset in the file for a particular contiguous request minus
         * what was satisfied in previous iteration
         * send_off    = offset the request needed by other processes in this iteration
         * req_len     = size corresponding to req_off
         * send_len    = size corresponding to send_off
         */

        /* first calculate what should be communicated */
        for (i = 0; i < nprocs; i++)
            recv_count[i] = recv_size[i] = send_size[i] = 0;

        off = off_list[m];
        max_size = MPL_MIN(step_size, max_end_loc - iter_st_off + 1);
        real_size = (int) MPL_MIN((off / stripe_size + 1) * stripe_size - off, end_loc - off + 1);

        for (i = 0; i < nprocs; i++) {
            if (my_req[i].count) {
                this_buf_idx[i] = buf_idx[i][send_curr_offlen_ptr[i]];
                for (j = send_curr_offlen_ptr[i]; j < my_req[i].count; j++) {
                    send_off = my_req[i].offsets[j];
                    send_len = my_req[i].lens[j];
                    if (send_off < iter_st_off + max_size) {
                        send_size[i] += send_len;
                    } else {
                        break;
                    }
                }
                send_curr_offlen_ptr[i] = j;
            }
            if (others_req[i].count) {
                recv_start_pos[i] = recv_curr_offlen_ptr[i];
                for (j = recv_curr_offlen_ptr[i]; j < others_req[i].count; j++) {
                    req_off = others_req[i].offsets[j];
                    req_len = others_req[i].lens[j];
                    if (req_off < iter_st_off + max_size) {
                        recv_count[i]++;
                        ADIOI_Assert((((ADIO_Offset) (uintptr_t) write_buf) + req_off - off) ==
                                     (ADIO_Offset) (uintptr_t) (write_buf + req_off - off));
                        MPI_Get_address(write_buf + req_off - off, &(others_req[i].mem_ptrs[j]));
                        recv_size[i] += req_len;
                    } else {
                        break;
                    }
                }
                recv_curr_offlen_ptr[i] = j;
            }
        }

        ADIOI_LUSTRE_W_Exchange_data(fd, buf, write_buf, flat_buf, offset_list,
                                     len_list, send_size, recv_size, off, real_size,
                                     recv_count, recv_start_pos,
                                     sent_to_proc, nprocs, myrank,
                                     contig_access_count,
                                     striping_info, others_req, send_buf_idx,
                                     curr_to_proc, done_to_proc, m,
                                     buftype_extent, this_buf_idx,
                                     &srt_off, &srt_len, &srt_num, error_code);

        if (*error_code != MPI_SUCCESS)
            goto over;

        iter_st_off += max_size;

        /* if there is no data to write for this iteration m */
        if (srt_num == 0)
            continue;

        /* lock ahead the file starting from off */
        ADIOI_LUSTRE_WR_LOCK_AHEAD(fd, striping_info[2], off, error_code);
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
             * [off, off+real_size). This assertion should never fail.
             */
            ADIOI_Assert(srt_off[i] < off + real_size && srt_off[i] >= off);

            ADIO_WriteContig(fd, write_buf + (srt_off[i] - off), srt_len[i],
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
}

/* Sets error_code to MPI_SUCCESS if successful, or creates an error code
 * in the case of error.
 */
static void ADIOI_LUSTRE_W_Exchange_data(ADIO_File fd, const void *buf,
                                         char *write_buf,
                                         ADIOI_Flatlist_node *flat_buf,
                                         ADIO_Offset *offset_list,
                                         ADIO_Offset *len_list, int *send_size,
                                         int *recv_size, ADIO_Offset real_off,
                                         int real_size, int *recv_count,
                                         int *start_pos,
                                         int *sent_to_proc, int nprocs,
                                         int myrank,
                                         int contig_access_count,
                                         int *striping_info,
                                         ADIOI_Access *others_req,
                                         ADIO_Offset *send_buf_idx,
                                         int *curr_to_proc, int *done_to_proc,
                                         int iter,
                                         MPI_Aint buftype_extent,
                                         ADIO_Offset *buf_idx,
                                         ADIO_Offset **srt_off, int **srt_len, int *srt_num,
                                         int *error_code)
{
    int i, j, k, nprocs_recv, nprocs_send, err;
    char **send_buf = NULL;
    MPI_Request *requests, *send_req;
    MPI_Datatype *recv_types;
    MPI_Status *statuses, status;
    int sum_recv, hole, check_hole;
    static size_t malloc_srt_num = 0;
    size_t send_total_size;
    static char myname[] = "ADIOI_LUSTRE_W_Exchange_data";

    /* calculate send receive metadata */
    *srt_num = 0;
    sum_recv = 0;
    nprocs_recv = 0;
    nprocs_send = 0;
    send_total_size = 0;
    for (i = 0; i < nprocs; i++) {
        *srt_num += recv_count[i];
        sum_recv += recv_size[i];
        if (recv_size[i])
            nprocs_recv++;
        if (send_size[i]) {
            nprocs_send++;
            send_total_size += send_size[i];
        }
    }

    /* create derived datatypes for recv */
    recv_types = (MPI_Datatype *) ADIOI_Malloc((nprocs_recv + 1) * sizeof(MPI_Datatype));
    /* +1 to avoid a 0-size malloc */

    j = 0;
    for (i = 0; i < nprocs; i++) {
        if (recv_size[i]) {
            ADIOI_Type_create_hindexed_x(recv_count[i],
                                         &(others_req[i].lens[start_pos[i]]),
                                         &(others_req[i].mem_ptrs[start_pos[i]]),
                                         MPI_BYTE, recv_types + j);
            /* absolute displacements; use MPI_BOTTOM in recv */
            MPI_Type_commit(recv_types + j);
            j++;
        }
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
            *srt_len = (int *) ADIOI_Malloc(*srt_num * sizeof(int));
            malloc_srt_num = *srt_num;
        }

        ADIOI_Heap_merge(others_req, recv_count, *srt_off, *srt_len, start_pos,
                         nprocs, nprocs_recv, *srt_num);

        /* (*srt_num) has been updated in heap_merge() such that (*srt_off) and
         * (*srt_len) were coalesced
         */
        hole = (*srt_num > 1);
    }

    /* check the hint for data sieving */
    if (fd->hints->ds_write != ADIOI_HINT_DISABLE && hole) {
        ADIO_ReadContig(fd, write_buf, real_size, MPI_BYTE, ADIO_EXPLICIT_OFFSET,
                        real_off, &status, &err);
        // --BEGIN ERROR HANDLING--
        if (err != MPI_SUCCESS) {
            *error_code = MPIO_Err_create_code(err,
                                               MPIR_ERR_RECOVERABLE,
                                               myname, __LINE__, MPI_ERR_IO, "**ioRMWrdwr", 0);
            ADIOI_Free(recv_types);
            return;
        }
        // --END ERROR HANDLING--

        /* Once read, holes have been filled and thus the number of
         * offset-length pairs, *srt_num, becomes one.
         */
        *srt_num = 1;
        if (*srt_off == NULL) {
            *srt_off = (ADIO_Offset *) ADIOI_Malloc(sizeof(ADIO_Offset));
            *srt_len = (int *) ADIOI_Malloc(sizeof(int));
            malloc_srt_num = 1;
        }
        (*srt_off)[0] = real_off;
        (*srt_len)[0] = real_size;
    }

    if (fd->atomicity) {
        /* bug fix from Wei-keng Liao and Kenin Coloma */
        requests = (MPI_Request *) ADIOI_Malloc((nprocs_send + 1) * sizeof(MPI_Request));
        send_req = requests;
    } else {
        requests = (MPI_Request *) ADIOI_Malloc((nprocs_send + nprocs_recv + 1) *
                                                sizeof(MPI_Request));
        /* +1 to avoid a 0-size malloc */

        /* post receives */
        j = 0;
        for (i = 0; i < nprocs; i++) {
            if (recv_size[i]) {
                MPI_Irecv(MPI_BOTTOM, 1, recv_types[j], i, ADIOI_COLL_TAG(i, iter), fd->comm,
                          requests + j);
                j++;
            }
        }
        send_req = requests + nprocs_recv;
    }

    /* post sends.
     * if buftype_is_contig, data can be directly sent from
     * user buf at location given by buf_idx. else use send_buf.
     */
    if (flat_buf->count == 1) {
        j = 0;
        for (i = 0; i < nprocs; i++)
            if (send_size[i]) {
                ADIOI_Assert(buf_idx[i] != -1);
                MPI_Issend(((char *) buf) + buf_idx[i], send_size[i],
                           MPI_BYTE, i, ADIOI_COLL_TAG(i, iter), fd->comm, send_req + j);
                j++;
            }
    } else if (nprocs_send) {
        /* buftype is not contig */
        send_buf = (char **) ADIOI_Malloc(nprocs * sizeof(char *));
        send_buf[0] = (char *) ADIOI_Malloc(send_total_size);
        for (i = 1; i < nprocs; i++)
            send_buf[i] = send_buf[i - 1] + send_size[i - 1];

        ADIOI_LUSTRE_Fill_send_buffer(fd, buf, flat_buf, send_buf, offset_list,
                                      len_list, send_size, send_req,
                                      sent_to_proc, nprocs, myrank,
                                      contig_access_count, striping_info,
                                      send_buf_idx, curr_to_proc, done_to_proc,
                                      iter, buftype_extent);
        /* the send is done in ADIOI_Fill_send_buffer */
    }

    /* bug fix from Wei-keng Liao and Kenin Coloma */
    if (fd->atomicity) {
        j = 0;
        for (i = 0; i < nprocs; i++) {
            MPI_Status wkl_status;
            if (recv_size[i]) {
                MPI_Recv(MPI_BOTTOM, 1, recv_types[j], i, ADIOI_COLL_TAG(i, iter), fd->comm,
                         &wkl_status);
                j++;
            }
        }
    }

    for (i = 0; i < nprocs_recv; i++)
        MPI_Type_free(recv_types + i);
    ADIOI_Free(recv_types);

#ifdef MPI_STATUSES_IGNORE
    statuses = MPI_STATUSES_IGNORE;
#else
    /* bug fix from Wei-keng Liao and Kenin Coloma */
    /* +1 to avoid a 0-size malloc */
    if (fd->atomicity) {
        statuses = (MPI_Status *) ADIOI_Malloc((nprocs_send + 1) * sizeof(MPI_Status));
    } else {
        statuses = (MPI_Status *) ADIOI_Malloc((nprocs_send + nprocs_recv + 1) *
                                               sizeof(MPI_Status));
    }
#endif

#ifdef NEEDS_MPI_TEST
    i = 0;
    if (fd->atomicity) {
        /* bug fix from Wei-keng Liao and Kenin Coloma */
        while (!i)
            MPI_Testall(nprocs_send, send_req, &i, statuses);
    } else {
        while (!i)
            MPI_Testall(nprocs_send + nprocs_recv, requests, &i, statuses);
    }
#else
    /* bug fix from Wei-keng Liao and Kenin Coloma */
    if (fd->atomicity)
        MPI_Waitall(nprocs_send, send_req, statuses);
    else
        MPI_Waitall(nprocs_send + nprocs_recv, requests, statuses);
#endif

#ifndef MPI_STATUSES_IGNORE
    ADIOI_Free(statuses);
#endif
    ADIOI_Free(requests);
    if (send_buf != NULL) {
        ADIOI_Free(send_buf[0]);
        ADIOI_Free(send_buf);
    }
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
        ADIOI_Assert((((ADIO_Offset)(uintptr_t)buf) + user_buf_idx) == (ADIO_Offset)(uintptr_t)((uintptr_t)buf + user_buf_idx)); \
        ADIOI_Assert(size_in_buf == (size_t)size_in_buf);               \
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
                                          ADIO_Offset *len_list, int *send_size,
                                          MPI_Request *requests,
                                          int *sent_to_proc, int nprocs,
                                          int myrank,
                                          int contig_access_count,
                                          int *striping_info,
                                          ADIO_Offset *send_buf_idx,
                                          int *curr_to_proc,
                                          int *done_to_proc, int iter, MPI_Aint buftype_extent)
{
    /* this function is only called if buftype is not contig */
    int i, p, flat_buf_idx, size;
    int flat_buf_sz, buf_incr, size_in_buf, jj, n_buftypes;
    ADIO_Offset off, len, rem_len, user_buf_idx;

    /* curr_to_proc[p] = amount of data sent to proc. p that has already
     * been accounted for so far
     * done_to_proc[p] = amount of data already sent to proc. p in
     * previous iterations
     * user_buf_idx = current location in user buffer
     * send_buf_idx[p] = current location in send_buf of proc. p
     */

    for (i = 0; i < nprocs; i++) {
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
    for (i = 0; i < contig_access_count; i++) {
        off = offset_list[i];
        rem_len = (ADIO_Offset) len_list[i];

        /*this request may span to more than one process */
        while (rem_len != 0) {
            len = rem_len;
            /* NOTE: len value is modified by ADIOI_Calc_aggregator() to be no
             * longer than the single region that processor "p" is responsible
             * for.
             */
            p = ADIOI_LUSTRE_Calc_aggregator(fd, off, &len, striping_info);

            if (send_buf_idx[p] < send_size[p]) {
                if (curr_to_proc[p] + len > done_to_proc[p]) {
                    if (done_to_proc[p] > curr_to_proc[p]) {
                        size = (int) MPL_MIN(curr_to_proc[p] + len -
                                             done_to_proc[p], send_size[p] - send_buf_idx[p]);
                        buf_incr = done_to_proc[p] - curr_to_proc[p];
                        ADIOI_BUF_INCR
                            ADIOI_Assert((curr_to_proc[p] + len - done_to_proc[p]) ==
                                         (unsigned) (curr_to_proc[p] + len - done_to_proc[p]));
                        buf_incr = (int) (curr_to_proc[p] + len - done_to_proc[p]);
                        ADIOI_Assert((done_to_proc[p] + size) ==
                                     (unsigned) (done_to_proc[p] + size));
                        curr_to_proc[p] = done_to_proc[p] + size;
                    ADIOI_BUF_COPY} else {
                        size = (int) MPL_MIN(len, send_size[p] - send_buf_idx[p]);
                        buf_incr = (int) len;
                        ADIOI_Assert((curr_to_proc[p] + size) ==
                                     (unsigned) ((ADIO_Offset) curr_to_proc[p] + size));
                        curr_to_proc[p] += size;
                    ADIOI_BUF_COPY}
                    if (send_buf_idx[p] == send_size[p]) {
                        MPI_Issend(send_buf[p], send_size[p], MPI_BYTE, p,
                                   ADIOI_COLL_TAG(p, iter), fd->comm, requests + jj);
                        jj++;
                    }
                } else {
                    ADIOI_Assert((curr_to_proc[p] + len) ==
                                 (unsigned) ((ADIO_Offset) curr_to_proc[p] + len));
                    curr_to_proc[p] += (int) len;
                    buf_incr = (int) len;
                ADIOI_BUF_INCR}
            } else {
                buf_incr = (int) len;
            ADIOI_BUF_INCR}
            off += len;
            rem_len -= len;
        }
    }
    for (i = 0; i < nprocs; i++)
        if (send_size[i])
            sent_to_proc[i] = curr_to_proc[i];
}

/* This function calls ADIOI_OneSidedWriteAggregation iteratively to
 * essentially pack stripes of data into the collective buffer and then
 * flush the collective buffer to the file when fully packed, repeating this
 * process until all the data is written to the file.
 */
static void ADIOI_LUSTRE_IterateOneSided(ADIO_File fd, const void *buf, int *striping_info,
                                         ADIO_Offset * offset_list, ADIO_Offset * len_list,
                                         int contig_access_count, int currentValidDataIndex,
                                         int count, int file_ptr_type, ADIO_Offset offset,
                                         ADIO_Offset start_offset, ADIO_Offset end_offset,
                                         ADIO_Offset firstFileOffset, ADIO_Offset lastFileOffset,
                                         MPI_Datatype buftype, int myrank, int *error_code)
{
    int i;
    int stripesPerAgg = fd->hints->cb_buffer_size / striping_info[0];
    if (stripesPerAgg == 0) {
        /* The striping unit is larger than the collective buffer size
         * therefore we must abort since the buffer has already been
         * allocated during the open.
         */
        FPRINTF(stderr, "Error: The collective buffer size %d is less "
                "than the striping unit size %d - the ROMIO "
                "Lustre one-sided write aggregation algorithm "
                "cannot continue.\n", fd->hints->cb_buffer_size, striping_info[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Based on the co_ratio the number of aggregators we can use is the number of
     * stripes used in the file times this co_ratio - each stripe is written by
     * co_ratio aggregators this information is contained in the striping_info.
     */
    int numStripedAggs = striping_info[2];

    int orig_cb_nodes = fd->hints->cb_nodes;
    fd->hints->cb_nodes = numStripedAggs;

    /* Declare ADIOI_OneSidedStripeParms here - these parameters will be locally managed
     * for this invocation of ADIOI_LUSTRE_IterateOneSided.  This will allow for concurrent
     * one-sided collective writes via multi-threading as well as multiple communicators.
     */
    ADIOI_OneSidedStripeParms stripeParms;
    stripeParms.stripeSize = striping_info[0];
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
    stripeParms.segmentLen = ((ADIO_Offset) numStripedAggs) * ((ADIO_Offset) (striping_info[0]));

    /* These arrays define the file offsets for the stripes for a given segment - similar
     * to the concept of file domains in GPFS, essentially file domeains for the segment.
     */
    ADIO_Offset *segment_stripe_start =
        (ADIO_Offset *) ADIOI_Malloc(numStripedAggs * sizeof(ADIO_Offset));
    ADIO_Offset *segment_stripe_end =
        (ADIO_Offset *) ADIOI_Malloc(numStripedAggs * sizeof(ADIO_Offset));

    /* Find the actual range of stripes in the file that have data in the offset
     * ranges being written -- skip holes at the front and back of the file.
     */
    int currentOffsetListIndex = 0;
    int fileSegmentIter = 0;
    int startingStripeWithData = 0;
    int foundStartingStripeWithData = 0;
    while (!foundStartingStripeWithData) {
        if (((startingStripeWithData + 1) * (ADIO_Offset) (striping_info[0])) > firstFileOffset)
            foundStartingStripeWithData = 1;
        else
            startingStripeWithData++;
    }

    ADIO_Offset currentSegementOffset =
        (ADIO_Offset) startingStripeWithData * (ADIO_Offset) (striping_info[0]);

    int numSegments =
        (int) ((lastFileOffset + (ADIO_Offset) 1 - currentSegementOffset) / stripeParms.segmentLen);
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

            int dataWrittenThisRound = 0;

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
                if ((segment_stripe_offset + (ADIO_Offset) (striping_info[0])) > lastFileOffset)
                    segment_stripe_end[i] = lastFileOffset;
                else
                    segment_stripe_end[i] =
                        segment_stripe_offset + (ADIO_Offset) (striping_info[0]) - (ADIO_Offset) 1;
                segment_stripe_offset += (ADIO_Offset) (striping_info[0]);
            }

            /* In the interest of performance for non-contiguous data with large offset lists
             * essentially modify the given offset and length list appropriately for this segment
             * and then pass pointers to the sections of the lists being used for this segment
             * to ADIOI_OneSidedWriteAggregation.  Remember how we have modified the list for this
             * segment, and then restore it appropriately after processing for this segment has
             * concluded, so it is ready for the next segment.
             */
            int segmentContigAccessCount = 0;
            int startingOffsetListIndex = -1;
            int endingOffsetListIndex = -1;
            ADIO_Offset startingOffsetAdvancement = 0;
            ADIO_Offset startingLenTrim = 0;
            ADIO_Offset endingLenTrim = 0;

            while (((offset_list[currentOffsetListIndex] +
                     ((ADIO_Offset) (len_list[currentOffsetListIndex])) - (ADIO_Offset) 1) <
                    segmentFirstFileOffset) && (currentOffsetListIndex < (contig_access_count - 1)))
                currentOffsetListIndex++;
            startingOffsetListIndex = currentOffsetListIndex;
            endingOffsetListIndex = currentOffsetListIndex;
            int offsetInSegment = 0;
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
                    dataWrittenThisRound += (int) len_list[currentOffsetListIndex];
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
                        dataWrittenThisRound += (int) len_list[currentOffsetListIndex];
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
                    dataWrittenThisRound -= (int) startingOffsetAdvancement;
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
                    dataWrittenThisRound -= (int) endingLenTrim;
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
                                               (ADIO_Offset *) &
                                               (offset_list[startingOffsetListIndex]),
                                               (ADIO_Offset *) &
                                               (len_list[startingOffsetListIndex]),
                                               segmentContigAccessCount,
                                               buf + totalDataWrittenLastRound, buftype,
                                               error_code, segmentFirstFileOffset,
                                               segmentLastFileOffset, currentValidDataIndex,
                                               segment_stripe_start, segment_stripe_end,
                                               &holeFoundThisRound, &stripeParms);
            } else {
                ADIOI_OneSidedWriteAggregation(fd,
                                               (ADIO_Offset *) &
                                               (offset_list[startingOffsetListIndex]),
                                               (ADIO_Offset *) &
                                               (len_list[startingOffsetListIndex]),
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
                    (ADIO_Offset) startingStripeWithData *(ADIO_Offset) (striping_info[0]);
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
