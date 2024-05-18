/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Copyright (C) 2024, Northwestern University
 *
 * This program tests ADIOI_Datatype_iscontig, a ROMIO's internal subroutine
 * and checks whether ROMIO can tell if the datatype is contiguous or not by
 * testing ADIOI_LUSTRE_Fill_send_buffer() on copying data from a
 * non-contiguous user buffer to an internal send buffer.
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <romioconf.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcpy() */
#include <unistd.h> /* getopt() */
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <mpi.h>

#include <adio.h> /* ADIOI_Datatype_iscontig() */

#define ERR \
    if (err != MPI_SUCCESS) { \
        int errorStringLen; \
        char errorString[MPI_MAX_ERROR_STRING]; \
        MPI_Error_string(err, errorString, &errorStringLen); \
        printf("Error at line %d: %s\n",__LINE__,errorString); \
        nerrs++; \
        goto err_out; \
    }

#ifndef ROMIO_INSIDE_MPICH
static void
usage(char *argv0)
{
    char *help =
    "Usage: %s [-hq] -f file_name]\n"
    "       [-h] Print this help\n"
    "       [-q] quiet mode\n"
    "        -f filename] output file name\n";
    fprintf(stderr, help, argv0);
}
#endif

#define LEN 4
#define NTIMES 2

/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv)
{
#ifdef ROMIO_INSIDE_MPICH
    return 0;
#else
    extern int optind;
    extern char *optarg;
    char filename[256];
    int i, j, k, err, nerrs=0, rank, mode, verbose=1;
    int *buf, sizes[2], start[2], is_contig, fd;
    int blocklen[4];
    MPI_Aint displace[4];
    MPI_Datatype dtype;
    MPI_File fh;
    MPI_Status status;
    MPI_Info info = MPI_INFO_NULL;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    filename[0] = '\0';

    /* get command-line arguments */
    while ((i = getopt(argc, argv, "hqf:")) != EOF)
        switch(i) {
            case 'q': verbose = 0;
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

    /* create a contiguous data type */
    sizes[0] = sizes[1]  = 10;
    start[0] = start[1]  = 0;
    err = MPI_Type_create_subarray(2, sizes, sizes, start, MPI_ORDER_C,
                                   MPI_INT, &dtype);
    ERR
    err = MPI_Type_commit(&dtype);
    ERR

    ADIOI_Datatype_iscontig(dtype, &is_contig);
    if (!is_contig)
        printf("Warning: MPI_Type_create_subarray datatype is not contiguous\n");

    err = MPI_Type_free(&dtype);
    ERR

    /* create a non-contiguous data type, a 4x4 2D array with rows unsorted */
    blocklen[0] = blocklen[1] = blocklen[2] = blocklen[3] = LEN;
    displace[0] = 0;
    displace[1] = 8;
    displace[2] = 4;
    displace[3] = 12;

    for (i=0; i<4; i++) displace[i] *= sizeof(int); /* unit in bytes */

    err = MPI_Type_create_hindexed(4, blocklen, displace, MPI_INT, &dtype);
    ERR
    err = MPI_Type_commit(&dtype);
    ERR

    ADIOI_Datatype_iscontig(dtype, &is_contig);
    if (!is_contig)
        printf("Warning: MPI_Type_create_hindexed datatype is not contiguous\n");

    buf = (int*) malloc(NTIMES * 4 * LEN * sizeof(int));
    for (i=0; i<4; i++) displace[i] /= sizeof(int); /* unit in int */

    for (k=0; k<NTIMES; k++) {
        int block = k * 4 * LEN;
        for (i=0; i<4; i++) {
            if (verbose) printf("buf[%2d]:",i+block);
            for (j=0; j<blocklen[i]; j++) {
                int indx = block + displace[i] + j;
                buf[block+i*LEN+j] = indx;
                if (verbose) printf(" %2d",buf[block+i*LEN+j]);
            }
            if (verbose) printf("\n");
        }
    }

    mode = MPI_MODE_CREATE | MPI_MODE_WRONLY;
    err = MPI_File_open(MPI_COMM_WORLD, filename, mode, info, &fh);
    ERR

    /* write to the file */
    err = MPI_File_write_all(fh, buf, NTIMES, dtype, &status);
    ERR

    err = MPI_Type_free(&dtype);
    ERR

    err = MPI_File_close(&fh);
    ERR

    /* read back from the file */
    fd = open(filename, O_RDONLY, 0400);
    read(fd, buf, NTIMES * 4 * LEN * sizeof(int));
    close(fd);

    /* check contents */
    for (i=0; i<NTIMES*4*LEN; i++) {
        if (buf[i] != i) {
            printf("Error: unexpected value in buf[%2d]=%d\n",i,buf[i]);
            nerrs++;
            break;
        }
    }
    free(buf);

err_out:
    MPI_Finalize();
    return (nerrs > 0);
#endif
}

