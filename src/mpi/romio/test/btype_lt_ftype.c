/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Copyright (C) 2024, Northwestern University
 *
 * This program tests collective write and calls using a noncontiguous fileview
 * datatype consisting of 2 large segments. The writes contains multiple
 * collective writes. The file access region of some writes falls entirely
 * within one of the two file segments of the fileview.
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcpy() */
#include <unistd.h> /* getopt() */

#include <mpi.h>

#define ERR \
    if (err != MPI_SUCCESS) { \
        int errorStringLen; \
        char errorString[MPI_MAX_ERROR_STRING]; \
        MPI_Error_string(err, errorString, &errorStringLen); \
        printf("Error at line %d: %s\n",__LINE__,errorString); \
        nerrs++; \
        goto err_out; \
    }

static void
usage(char *argv0)
{
    char *help =
    "Usage: %s [-hq | -l len | -n num] -f file_name\n"
    "       [-h] Print this help\n"
    "       [-q] quiet mode\n"
    "       [-l len] length of local X and Y dimension sizes\n"
    "       [-n num] number of writes (must <= 4, default 2)\n"
    "        -f filename: output file name\n";
    fprintf(stderr, help, argv0);
}

/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv)
{
    extern int optind;
    extern char *optarg;
    char filename[256];
    size_t i, j, k, n;
    int err, nerrs=0, rank, nprocs, mode, verbose=1, ntimes, len;
    int *rbuf[4], gsizes[2], subsizes[2], starts[2], lsizes[2];
    int *buf=NULL, ftype_size, btype_size, max_nerrs;
    double timing, max_timing;
    MPI_Aint lb, displace[2], fExtent, bExtent;
    MPI_Datatype bufType, fileType;
    MPI_File fh;
    MPI_Status status;
    MPI_Info info;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ntimes = 2;
    len = 100;  /* default dimension size */
    filename[0] = '\0';

    /* get command-line arguments */
    while ((i = getopt(argc, argv, "hql:n:f:")) != EOF)
        switch(i) {
            case 'q': verbose = 0;
                      break;
            case 'n': ntimes = atoi(optarg);
                      break;
            case 'l': len = atoi(optarg);
                      break;
            case 'f': strcpy(filename, optarg);
                      break;
            case 'h':
            default:  if (rank==0) usage(argv[0]);
                      MPI_Finalize();
                      return 1;
        }

    if (filename[0] == '\0') {
        if (rank==0) usage(argv[0]);
        MPI_Finalize();
        return 1;
    }
    if (ntimes > 4) ntimes = 4;

    if (verbose && rank == 0) {
        printf("Creating a fileview datatype consisting of 2 blocks\n");
        printf("Each block is of size %d x %d (int)= %zd\n",
               len, len, sizeof(int)*len*len);
        printf("Gap between two consecutive blocks is %d x %d ints\n",
               len*len, nprocs -1);
    }

    /* create a user buffer datatype, a subarray, with ghost cells */
    gsizes[0]   = len + 2; /* global array size */
    gsizes[1]   = len + 2; /* ghost cells of size 2 */
    starts[0]   = 1;
    starts[1]   = 1;
    subsizes[0] = len;
    subsizes[1] = len;
    err = MPI_Type_create_subarray(2, gsizes, subsizes, starts, MPI_ORDER_C,
                                   MPI_INT, &bufType);
    ERR
    err = MPI_Type_commit(&bufType); ERR

    /* allocate I/O buffer */
    MPI_Type_size(bufType, &btype_size);
    lb = 0;
    MPI_Type_get_extent(bufType, &lb, &bExtent);
    if (verbose && rank == 0)
        printf("buffer type size = %d extent = %ld\n", btype_size, bExtent);

    buf = (int*) calloc(bExtent, 1);
    k = 0;
    for (i=1; i<len+1; i++)
        for (j=1; j<len+1; j++) {
            buf[i*(len+2)+j] = (k + 17 + rank) % 2147483647;
            k++;
        }

    /* create fileview data type. It consists of 2 noncontiguous file segments
     * and each segment is of size == 2 * btype_size
     */
    lsizes[0] = btype_size * 2;
    lsizes[1] = btype_size * 2;
    displace[0] = rank * btype_size * 2;
    displace[1] = displace[0] + nprocs * btype_size * 2;
    err = MPI_Type_create_hindexed(2, lsizes, displace, MPI_BYTE, &fileType);
    ERR
    err = MPI_Type_commit(&fileType); ERR

    MPI_Type_size(fileType, &ftype_size);
    lb = 0;
    MPI_Type_get_extent(fileType, &lb, &fExtent);
    if (verbose && rank == 0)
        printf("file   type size = %d extent = %ld\n", ftype_size, fExtent);

    /* force to use collective subroutines */
    MPI_Info_create(&info);
    MPI_Info_set(info, "romio_cb_write", "enable");
    MPI_Info_set(info, "romio_cb_read", "enable");

    /* open file */
    mode = MPI_MODE_CREATE | MPI_MODE_RDWR;
    err = MPI_File_open(MPI_COMM_WORLD, filename, mode, info, &fh); ERR

    err = MPI_Info_free(&info); ERR

    /* set the file view */
    err = MPI_File_set_view(fh, 0, MPI_BYTE, fileType, "native", MPI_INFO_NULL);
    ERR

    err = MPI_Type_free(&fileType); ERR

    /* write to the file */
    MPI_Barrier(MPI_COMM_WORLD);
    timing = MPI_Wtime();
    for (i=0; i<ntimes; i++) {
        err = MPI_File_write_all(fh, buf, 1, bufType, &status);
        ERR
    }
    free(buf);

    /* read back */
    err = MPI_File_seek(fh, 0, MPI_SEEK_SET); ERR
    for (i=0; i<ntimes; i++) {
        rbuf[i] = (int*) calloc(bExtent, 1);
        err = MPI_File_read_all(fh, rbuf[i], 1, bufType, &status);
        ERR
    }
    timing = MPI_Wtime() - timing;

    err = MPI_Type_free(&bufType); ERR

    /* check read contents */
    for (n=0; n<ntimes; n++) {
        k = 0;
        for (i=1; i<len+1; i++) {
            for (j=1; j<len+1; j++) {
                int exp = (k + 17 + rank) % 2147483647;
                if (rbuf[n][i*(len+2)+j] != exp) {
                    printf("Error: rbuf[%zd][%zd] expect %d but got %d\n", n, i*(len+2)+j, exp, rbuf[n][i*(len+2)+j]);
                    nerrs++;
                    break;
                }
                k++;
            }
        }
    }

    err = MPI_File_close(&fh); ERR
    for (n=0; n<ntimes; n++) free(rbuf[n]);

    MPI_Allreduce(&nerrs, &max_nerrs, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Reduce(&timing, &max_timing, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (max_nerrs == 0 && rank == 0)
        printf("Time of collective write and read = %.2f sec\n", max_timing);

err_out:
    MPI_Finalize();
    return (nerrs > 0);
}

