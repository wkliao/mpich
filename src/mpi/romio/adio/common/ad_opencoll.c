/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "adio.h"

static int construct_aggr_list(ADIO_File fd, int *error_code);

/* Generic version of a "collective open".  Assumes a "real" underlying
 * file system (meaning no wonky consistency semantics like NFS).
 *
 * optimization: by having just one process create a file, close it,
 * then have all N processes open it, we can possibly avoid contention
 * for write locks on a directory for some file systems.
 *
 * Happy side-effect: exclusive create (error if file already exists)
 * just falls out
 *
 * Note: this is not a "scalable open" (c.f. "The impact of file systems
 * on MPI-IO scalability").
 */

enum {
    BLOCKSIZE = 0,
    STRIPE_SIZE,
    STRIPE_FACTOR,
    START_IODEVICE,
    STAT_ITEMS
} file_stats;


/* generate an MPI datatype describing the members of the ADIO_File struct that
 * we want to ensure all processes have.  In deferred open, aggregators will
 * open the file and possibly read layout and other information.
 * non-aggregators will skip the open, but still need to know how the file is
 * being treated and what optimizations to apply */

static MPI_Datatype make_stats_type(ADIO_File fd)
{
    int lens[STAT_ITEMS];
    MPI_Aint offsets[STAT_ITEMS];
    MPI_Datatype types[STAT_ITEMS];
    MPI_Datatype newtype;

    lens[BLOCKSIZE] = 1;
    MPI_Get_address(&fd->blksize, &offsets[BLOCKSIZE]);
    types[BLOCKSIZE] = MPI_LONG;

    lens[STRIPE_SIZE] = lens[STRIPE_FACTOR] = lens[START_IODEVICE] = 1;
    types[STRIPE_SIZE] = types[STRIPE_FACTOR] = types[START_IODEVICE] = MPI_INT;
    MPI_Get_address(&fd->hints->striping_unit, &offsets[STRIPE_SIZE]);
    MPI_Get_address(&fd->hints->striping_factor, &offsets[STRIPE_FACTOR]);
    MPI_Get_address(&fd->hints->start_iodevice, &offsets[START_IODEVICE]);


    MPI_Type_create_struct(STAT_ITEMS, lens, offsets, types, &newtype);
    MPI_Type_commit(&newtype);
    return newtype;

}

void ADIOI_GEN_OpenColl(ADIO_File fd, int rank, int access_mode, int *error_code)
{
    char value[MPI_MAX_INFO_VAL + 1];
    int i, orig_amode_excl, orig_amode_wronly;
    MPI_Comm tmp_comm;
    MPI_Datatype stats_type;    /* deferred open: some processes might not
                                 * open the file, so we'll exchange some
                                 * information with those non-aggregators */

    orig_amode_excl = access_mode;

    if ((access_mode & ADIO_CREATE) || fd->hints->ranklist == NULL) {
        /* fd->hints->ranklist is NULL only when the file is on Lustre and
         * cb_config_list hint is not set in the user info object. Its
         * construction has been delayed from ADIO_Open() until here. We need
         * to obtain Lustre file stripe count on root process, in order to
         * construct aggregator rank list, no matter if create or not for
         * Lustre.
         */
        int root = (fd->hints->ranklist == NULL) ? 0 : fd->hints->ranklist[0];

        if (rank == root) {
            /* remove delete_on_close flag if set */
            if (access_mode & ADIO_DELETE_ON_CLOSE)
                fd->access_mode = access_mode ^ ADIO_DELETE_ON_CLOSE;
            else
                fd->access_mode = access_mode;

            tmp_comm = fd->comm;
            fd->comm = MPI_COMM_SELF;
            (*(fd->fns->ADIOI_xxx_Open)) (fd, error_code);
            fd->comm = tmp_comm;
            MPI_Bcast(error_code, 1, MPI_INT, root, fd->comm);
            /* if no error, close the file and reopen normally below */
            if (*error_code == MPI_SUCCESS)
                (*(fd->fns->ADIOI_xxx_Close)) (fd, error_code);

            fd->access_mode = access_mode;      /* back to original */
        } else
            MPI_Bcast(error_code, 1, MPI_INT, root, fd->comm);

        if (*error_code != MPI_SUCCESS) {
            return;
        } else {
            /* turn off CREAT (and EXCL if set) for real multi-processor open */
            access_mode ^= ADIO_CREATE;
            if (access_mode & ADIO_EXCL)
                access_mode ^= ADIO_EXCL;
        }

        if (fd->hints->ranklist == NULL) {
            /* Once the Lustre file stripe count, fd->hints->striping_factor,
             * is set, we use it to construct the I/O aggregator rank list,
             * fd->hints->ranklist[].
             */
            construct_aggr_list(fd, error_code);
            if (*error_code != MPI_SUCCESS)
                return;
        }
    }
/* TODO:
 * When deferred_open is true: (only aggregators open the file)
   * File create:
     1. root creates the file
        a. root obtains striping info
        b. root closes file
     2. All processes determine cb_nodes  <-----
     3. Only aggregators open the file
     4. all processes sync and set file striping info
        a. root bcasts striping info
        b. all processes set hints
   * File open:
     1. root opens the file
        a. root obtains striping info
        b. root closes file
     2. All processes determine cb_nodes  <-----
     3. Only aggregators open the file
     4. all processes sync and set file striping info
        a. root bcasts striping info
        b. all processes set hints

 * When deferred_open is false (all processes open the file)
   * File create:
     1. root creates the file
        a. root obtains striping info
        b. root closes file
     2. All processes open the file
        a. MUST determine cb_nodes, I/O aggregators  <-----
     3. all processes sync and set file striping info
        a. root bcasts striping info
        b. all processes set hints
   * File open:
     1. All processes open the file
        a. MUST determine cb_nodes, I/O aggregators  <-----
     3. All processes sync and set file striping info
        a. root bcasts striping info
        b. all processes set hints
*/

    fd->blksize = 1024 * 1024 * 4;      /* this large default value should be good for
                                         * most file systems.  any ROMIO driver is free
                                         * to stat the file and find an optimial value */

    /* if we are doing deferred open, non-aggregators should return now */
    if (fd->hints->deferred_open) {
        if (!(fd->is_agg)) {
            /* we might have turned off EXCL for the aggregators.
             * restore access_mode that non-aggregators get the right
             * value from get_amode */
            fd->access_mode = orig_amode_excl;
            /* In file-system specific open, a driver might collect some
             * information via stat().  Deferred open means not every process
             * participates in fs-specific open, but they all participate in
             * this open call.  Broadcast a bit of information in case
             * lower-level file system driver (e.g. 'bluegene') collected it
             * (not all do)*/
            stats_type = make_stats_type(fd);
            /* in this branch the non-aggregators are returning early.
             * MPI_Bcast here matches the MPI_Bcast from the aggregators after
             * they open the file and find out striping information */
            MPI_Bcast(MPI_BOTTOM, 1, stats_type, fd->hints->ranklist[0], fd->comm);
            ADIOI_Assert(fd->blksize > 0);
            /* some file systems (e.g. lustre) will inform the user via the
             * info object about the file configuration.  deferred open,
             * though, skips that step for non-aggregators.  we do the
             * info-setting here */
            snprintf(value, sizeof(value), "%d", fd->hints->striping_unit);
            ADIOI_Info_set(fd->info, "striping_unit", value);

            snprintf(value, sizeof(value), "%d", fd->hints->striping_factor);
            ADIOI_Info_set(fd->info, "striping_factor", value);

            snprintf(value, sizeof(value), "%d", fd->hints->start_iodevice);
            ADIOI_Info_set(fd->info, "start_iodevice", value);

            *error_code = MPI_SUCCESS;
            MPI_Type_free(&stats_type);
            return;
        }
    }

/* For writing with data sieving, a read-modify-write is needed. If
   the file is opened for write_only, the read will fail. Therefore,
   if write_only, open the file as read_write, but record it as write_only
   in fd, so that get_amode returns the right answer. */

    /* observation from David Knaak: file systems that do not support data
     * sieving do not need to change the mode */

    orig_amode_wronly = access_mode;
    if ((access_mode & ADIO_WRONLY) && ADIO_Feature(fd, ADIO_DATA_SIEVING_WRITES)) {
        access_mode = access_mode ^ ADIO_WRONLY;
        access_mode = access_mode | ADIO_RDWR;
    }
    fd->access_mode = access_mode;

    (*(fd->fns->ADIOI_xxx_Open)) (fd, error_code);

    /* if error, may be it was due to the change in amode above.
     * therefore, reopen with access mode provided by the user. */
    fd->access_mode = orig_amode_wronly;
    if (*error_code != MPI_SUCCESS)
        (*(fd->fns->ADIOI_xxx_Open)) (fd, error_code);

    /* if we turned off EXCL earlier, then we should turn it back on */
    if (fd->access_mode != orig_amode_excl)
        fd->access_mode = orig_amode_excl;

    /* broadcast information to all processes in
     * communicator, not just those who participated in open */

    stats_type = make_stats_type(fd);
    MPI_Bcast(MPI_BOTTOM, 1, stats_type, fd->hints->ranklist[0], fd->comm);
    MPI_Type_free(&stats_type);
    /* file domain code will get terribly confused in a hard-to-debug way if
     * gpfs blocksize not sensible */
    ADIOI_Assert(fd->blksize > 0);

    /* set file striping hints */
    snprintf(value, sizeof(value), "%d", fd->hints->striping_unit);
    ADIOI_Info_set(fd->info, "striping_unit", value);

    snprintf(value, sizeof(value), "%d", fd->hints->striping_factor);
    ADIOI_Info_set(fd->info, "striping_factor", value);

    snprintf(value, sizeof(value), "%d", fd->hints->start_iodevice);
    ADIOI_Info_set(fd->info, "start_iodevice", value);

    /* for deferred open: this process has opened the file (because if we are
     * not an aggregaor and we are doing deferred open, we returned earlier)*/
    fd->is_open = 1;

    /* sync optimization: we can omit the fsync() call if we do no writes */
    fd->dirty_write = 0;

    /* add to fd->info the hint "aggr_list", list of aggregators' rank IDs */
    value[0] = '\0';
    for (i = 0; i < fd->hints->cb_nodes; i++) {
        char str[16];
        if (i == 0)
            snprintf(str, sizeof(str), "%d", fd->hints->ranklist[i]);
        else
            snprintf(str, sizeof(str), " %d", fd->hints->ranklist[i]);
        if (strlen(value) + strlen(str) > MPI_MAX_INFO_VAL)
            break;
        strcat(value, str);
    }
    ADIOI_Info_set(fd->info, "aggr_list", value);
}

/*----< construct_aggr_list() >----------------------------------------------*/
/* Allocate and construct fd->hints->ranklist[].
 * Overwrite fd->hints->cb_nodes and set hint cb_nodes.
 * Set fd->is_agg
 */
#define NUM_STRIPING_INFO 5
static int construct_aggr_list(ADIO_File fd, int *error_code)
{
    int i, j, rank, nprocs, num_aggr, my_procname_len;
    int msg[NUM_STRIPING_INFO], num_osts, overstriping;
    int *all_procname_lens = NULL;
    char *value, my_procname[MPI_MAX_PROCESSOR_NAME];
    char **all_procnames = NULL;
    static char myname[] = "ADIO_OPENCOLL construct_aggr_list";

    MPI_Comm_size(fd->comm, &nprocs);
    MPI_Comm_rank(fd->comm, &rank);

    /* at this moment, only root process has obtained the file striping
      * settings and num_osts.
     */
    if (rank == 0) {
        num_osts = fd->hints->fs_hints.lustre.num_osts;
        /* When number of MPI processes is less than num_osts */
        if (num_osts > nprocs) {
            /* Find the max number less than nprocs that divides num_osts.
             * An naive way is:
             *     num_aggr = nprocs;
             *     while (num_osts % num_aggr > 0)
             *         num_aggr--;
             * Below is equivalent, but faster.
             */
            int divisor = 2;
            num_aggr = 1;
            /* try to divide */
            while (num_osts >= divisor * divisor) {
                if ((num_osts % divisor) == 0) {
                    if (num_osts / divisor <= nprocs) {
                        /* The value is found ! */
                        num_aggr = num_osts / divisor;
                        break;
                    }
                    /* if divisor is less than nprocs, divisor is a solution, but
                     * it is not sure that it is the best one
                     */
                    else if (divisor <= nprocs)
                        num_aggr = divisor;
                }
                divisor++;
            }
        }
        else { /* num_osts <= nprocs */
            if (fd->hints->cb_nodes == 0) /* hint cb_nodes is not set by user */
                num_aggr = fd->hints->striping_factor;
            else if (num_osts > fd->hints->cb_nodes)
                /* num_osts > cb_nodes */
                num_aggr = num_osts;
            else {
                /* num_osts <= cb_nodes */
                if (fd->hints->cb_nodes % num_osts == 0)
                    /* cb_nodes is a multiple of num_osts */
                    num_aggr = fd->hints->cb_nodes;
                else {
                    /* find the biggest multiple of num_osts that is less
                     * than cb_nodes
                     */
                    num_aggr = fd->hints->cb_nodes / num_osts;
                    num_aggr *= num_osts;
                }
            }
        }
        msg[0] = fd->hints->striping_factor;
        msg[1] = fd->hints->striping_unit;
        msg[2] = fd->hints->start_iodevice;
        msg[3] = fd->hints->fs_hints.lustre.num_osts;
        msg[4] = num_aggr;

        /* TODO: the above setting for num_aggr is for collective writes. Reads
         * should be the number of nodes.
         */
    }

    /* sync file striping info among all processes */
    MPI_Bcast(msg, NUM_STRIPING_INFO, MPI_INT, 0, fd->comm);
    fd->hints->striping_factor          = msg[0];
    fd->hints->striping_unit            = msg[1];
    fd->hints->start_iodevice           = msg[2];
    fd->hints->fs_hints.lustre.num_osts = msg[3];
    num_aggr                            = msg[4];

    /* overwrite cb_nodes */
    fd->hints->cb_nodes = num_aggr;

    /* set file striping hints */
    value = (char *) ADIOI_Malloc(12);  /* int has at most 12 digits */
    sprintf(value, "%d", fd->hints->cb_nodes);
    ADIOI_Info_set(fd->info, "cb_nodes", value);
    sprintf(value, "%d", fd->hints->striping_factor);
    ADIOI_Info_set(fd->info, "striping_factor", value);
    sprintf(value, "%d", fd->hints->striping_unit);
    ADIOI_Info_set(fd->info, "striping_unit", value);
    sprintf(value, "%d", fd->hints->start_iodevice);
    ADIOI_Info_set(fd->info, "start_iodevice", value);
    sprintf(value, "%d", fd->hints->fs_hints.lustre.num_osts);
    ADIOI_Info_set(fd->info, "lustre_num_osts", value);
    ADIOI_Free(value);

    /* ranklist[] contains the MPI ranks of I/O aggregators */
    fd->hints->ranklist = (int *) ADIOI_Malloc(num_aggr * sizeof(int));
    if (fd->hints->ranklist == NULL) {
        *error_code = MPIO_Err_create_code(*error_code,
                                           MPIR_ERR_RECOVERABLE, myname,
                                           __LINE__, MPI_ERR_OTHER,
                                           "**nomem2", 0);
        return 0;
    }

    /* collect info about compute nodes in order to select aggregators */
    MPI_Get_processor_name(my_procname, &my_procname_len);
    my_procname_len++; /* to include terminate null character */

    if (rank == 0) {
        /* process 0 collects all procnames */
        all_procnames = (char **) ADIOI_Malloc(nprocs * sizeof(char *));
        if (all_procnames == NULL) {
            *error_code = MPIO_Err_create_code(*error_code,
                                               MPIR_ERR_RECOVERABLE, myname,
                                               __LINE__, MPI_ERR_OTHER,
                                               "**nomem2", 0);
            return 0;
        }

        all_procname_lens = (int *) ADIOI_Malloc(nprocs * sizeof(int));
        if (all_procname_lens == NULL) {
            ADIOI_Free(all_procnames);
            *error_code = MPIO_Err_create_code(*error_code,
                                               MPIR_ERR_RECOVERABLE, myname,
                                                __LINE__, MPI_ERR_OTHER,
                                                "**nomem2", 0);
            return 0;
        }
    }
    /* gather lengths first */
    MPI_Gather(&my_procname_len, 1, MPI_INT, all_procname_lens, 1, MPI_INT, 0,
               fd->comm);

    if (rank == 0) {
        int *disp;
        size_t alloc_size = 0;

        for (i = 0; i < nprocs; i++)
            alloc_size += all_procname_lens[i];

        all_procnames[0] = (char *) ADIOI_Malloc(alloc_size);
        if (all_procnames[0] == NULL) {
            ADIOI_Free(all_procname_lens);
            ADIOI_Free(all_procnames);
            *error_code = MPIO_Err_create_code(*error_code,
                                               MPIR_ERR_RECOVERABLE, myname,
                                               __LINE__, MPI_ERR_OTHER,
                                               "**nomem2", 0);
            return 0;
        }

        /* Construct displacement array for the MPI_Gatherv, as each process
         * may have a different length for its process name.
         */
        disp = (int *) ADIOI_Malloc(nprocs * sizeof(int));
        disp[0] = 0;
        for (i = 1; i < nprocs; i++) {
            all_procnames[i] = all_procnames[i - 1] + all_procname_lens[i - 1];
            disp[i] = disp[i - 1] + all_procname_lens[i - 1];
        }

        /* gather all process names */
        MPI_Gatherv(my_procname, my_procname_len, MPI_CHAR,
                    all_procnames[0], all_procname_lens, disp, MPI_CHAR, 0,
                    fd->comm);

        ADIOI_Free(disp);
        ADIOI_Free(all_procname_lens);
    } else
        /* send process name to root */
        MPI_Gatherv(my_procname, my_procname_len, MPI_CHAR,
                    NULL, NULL, NULL, MPI_CHAR, 0, fd->comm);

    /* root constructs ranklist[] */
    if (rank == 0) {
        int n, last, num_nodes;
        char **node_names;
        int *nprocs_per_node;
        int **ranks_per_node;
        int *node_ids;

        /* number of MPI processes running on each node */
        nprocs_per_node = (int *) ADIOI_Malloc(nprocs * sizeof(int));

        /* compute node IDs of MPI processes */
        node_ids = (int *) ADIOI_Malloc(nprocs * sizeof(int));

        /* array of unique host names (compute nodes) */
        node_names = (char **) ADIOI_Malloc(nprocs * sizeof(char *));

        /* calculate nprocs_per_node[] and node_ids[] */
        last = 0;
        num_nodes = 0;
        for (i = 0; i < nprocs; i++) {
            n = last;
            for (j = 0; j < num_nodes; j++) {
                /* check if [i] has already appeared in [] */
                if (!strcmp(all_procnames[i], node_names[n])) { /* found */
                    node_ids[i] = n;
                    nprocs_per_node[n]++;
                    break;
                }
                n = (n == num_nodes - 1) ? 0 : n + 1;
            }
            if (j < num_nodes)  /* found, next iteration, start with node n */
                last = n;
            else {      /* not found, j == num_nodes, add a new node */
                node_names[j] = ADIOI_Strdup(all_procnames[i]);
                nprocs_per_node[j] = 1;
                node_ids[i] = j;
                last = j;
                num_nodes++;
            }
        }
        /* num_nodes is now the number of compute nodes (unique node names) */
        fd->num_nodes = num_nodes;

        for (i = 0; i < num_nodes; i++)
            ADIOI_Free(node_names[i]);
        ADIOI_Free(node_names);
        ADIOI_Free(all_procnames[0]);
        ADIOI_Free(all_procnames);

        /* construct rank IDs of MPI processes running on each node */
        ranks_per_node = (int **) ADIOI_Malloc(num_nodes * sizeof(int *));
        ranks_per_node[0] = (int *) ADIOI_Malloc(nprocs * sizeof(int));
        for (i = 1; i < num_nodes; i++)
            ranks_per_node[i] = ranks_per_node[i - 1] + nprocs_per_node[i - 1];
        for (i = 0; i < num_nodes; i++)
            nprocs_per_node[i] = 0;

        /* populate ranks_per_node[] */
        for (i = 0; i < nprocs; i++) {
            n = node_ids[i];
            ranks_per_node[n][nprocs_per_node[n]] = i;
            nprocs_per_node[n]++;
        }
        ADIOI_Free(node_ids);

        overstriping = (fd->hints->striping_factor > num_osts) ? 1 : 0;
        if ((!overstriping && num_osts >= num_nodes) ||
            ( overstriping && num_aggr >= num_nodes)) {
            /* When number of OSTs is more than number of compute nodes,
             * pick processes (spread evenly) from each node to be aggregators.
             */
            int num = num_aggr / num_nodes;

            /* number of processes selected in node n */
            num = (num_aggr % num_nodes) ? num + 1 : num;
            for (i = 0; i < num_aggr; i++) {
                int stride;
                n = i % num_nodes;      /* node ID */
                if (num >= nprocs_per_node[n])
                    stride = 1;
                else
                    stride = nprocs_per_node[n] / num;
                fd->hints->ranklist[i] = ranks_per_node[n][i / num_nodes * stride];
            }
        } else {
            /* When number of OSTs is less than number of compute nodes,
             * select evenly spread compute nodes and pick first process of
             * selected node to be aggregators.
             */
            if (!overstriping) {
                int k, avg = num_aggr / num_osts;
                int stride = num_nodes / num_osts;
                for (i = 0; i < num_aggr; i++) {
                    j = (i * stride) % num_nodes;
                    k = (i / num_osts) * (nprocs_per_node[j] / avg);
                    fd->hints->ranklist[i] = ranks_per_node[j][k];
                }
            } else {
                int stride = num_nodes / num_aggr;
                for (i = 0; i < num_aggr; i++)
                    fd->hints->ranklist[i] = ranks_per_node[i * stride][0];
            }
        }

        /* TODO: we can keep these two arrays in case for dynamic construction
         * of fd->hints->ranklist[], such as in group-cyclic file domain
         * assignment method, used in each collective write call.
         */
        ADIOI_Free(nprocs_per_node);
        ADIOI_Free(ranks_per_node[0]);
        ADIOI_Free(ranks_per_node);
    }

    MPI_Bcast(fd->hints->ranklist, fd->hints->cb_nodes, MPI_INT, 0, fd->comm);

    /* check whether this process is selected as an I/O aggregator */
    fd->is_agg = 0;
    for (i = 0; i < num_aggr; i++) {
        if (rank == fd->hints->ranklist[i]) {
            fd->is_agg = 1;
            break;
        }
    }

    return 0;
}

