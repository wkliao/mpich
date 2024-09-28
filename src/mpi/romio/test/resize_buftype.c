/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Copyright (C) 2024, Northwestern University
 *
 * This program tests collective write and calls using a noncontiguous fileview
 * datatype consisting of 2 large segments. The writes contains multiple
 * collective writes. The file access region of some writes falls entirely
 * within one of the two file segments of the fileview. The user buffer is
 * contiguous, but buftype is made noncontiguous on purpose, by calling
 * MPI_Type_create_resized().
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
    size_t i, k, n;
    int err, nerrs=0, rank, nprocs, mode, verbose=1, ntimes, len;
    int *rbuf[4], lsizes[2];
    int *buf=NULL, ftype_size, btype_size, max_nerrs;
    double timing, max_timing;
    MPI_Aint lb, displace[2], fExtent, bExtent;
    MPI_Datatype type_contig, bufType, fileType;
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
        printf("Each block is of size %d (int)= %zd\n",
               len, sizeof(int)*len);
        printf("Gap between two consecutive blocks is %d x %d ints\n",
               len, nprocs -1);
    }

    /* create a user buffer datatype, a contiguous datatype */
    err = MPI_Type_contiguous(len, MPI_INT, &type_contig); ERR
    err = MPI_Type_commit(&type_contig); ERR

    /* resize it to make it noncontiguous */
    bExtent = len * sizeof(int) + 10;
    err = MPI_Type_create_resized(type_contig, 0, bExtent, &bufType); ERR

    err = MPI_Type_commit(&bufType); ERR
    err = MPI_Type_free(&type_contig); ERR

    /* allocate I/O buffer */
    MPI_Type_size(bufType, &btype_size);
    MPI_Type_get_extent(bufType, &lb, &bExtent);
    if (verbose && rank == 0)
        printf("buffer type size=%d lb=%ld extent=%ld\n",btype_size,lb,bExtent);

    buf = (int*) calloc(btype_size, 1);
    k = 0;
    for (i=0; i<len; i++) {
        buf[i] = (k + 17 + rank) % 2147483647;
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

    err = MPI_Type_size(fileType, &ftype_size); ERR
    lb = 0;
    err = MPI_Type_get_extent(fileType, &lb, &fExtent); ERR
    if (verbose)
        printf("%d: file type size=%d extent=%ld\n", rank,ftype_size,fExtent);

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
        for (i=0; i<len; i++) {
            int exp = (k + 17 + rank) % 2147483647;
            if (rbuf[n][i] != exp) {
                printf("Error: rbuf[%zd][%zd] expect %d but got %d\n", n, i, exp, rbuf[n][i]);
                nerrs++;
                break;
            }
            k++;
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

