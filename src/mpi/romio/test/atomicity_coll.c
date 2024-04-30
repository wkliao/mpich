/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Copyright (C) 2024, Northwestern University
 *
 * This program tests collective write for when atomicity is enabled.
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcpy() */
#include <unistd.h> /* getopt() */

#include <mpi.h>

#define LEN 2048

#define MPI_CHECK(err) \
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
    "Usage: %s [-hvrw | -l num ] -f file_name\n"
    "       [-h] Print this help\n"
    "       [-v] verbose mode\n"
    "       [-w] performs write only (default: both write and read)\n"
    "       [-r] performs read  only (default: both write and read)\n"
    "       [-l num] number of columns in each global variable (default: %d)\n"
    "        -f file_name: output file name\n";
    fprintf(stderr, help, argv0, LEN);
}

/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv)
{
    extern int optind;
    extern char *optarg;
    char filename[256];
    int i, err, nerrs=0, rank, nprocs, mode, verbose=0, len;
    int do_write, do_read, *buf;
    MPI_File fh;
    MPI_Offset offset;
    MPI_Info info = MPI_INFO_NULL;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    len      = LEN;
    do_write = 1;
    do_read  = 1;
    filename[0] = '\0';

    /* get command-line arguments */
    while ((i = getopt(argc, argv, "hvwrl:f:")) != EOF)
        switch(i) {
            case 'v': verbose = 1;
                      break;
            case 'l': len = atoi(optarg);
                      break;
            case 'w': do_read = 0;
                      break;
            case 'r': do_write = 0;
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

#define CB_BUFFER_SIZE "2048"
#define CB_NODES "4"
#define STRIPING_UNIT "1024"
#define STRIPING_FACTOR "4"

    if (verbose && rank == 0) {
        printf("Number of MPI processes = %d\n",nprocs);
        if (do_write) printf("Perform collective write\n");
        if (do_read) printf("Perform collective read\n");
        printf("Each  process writes a contiguous space of size %d integers\n",len);
        printf("ROMIO hint set: cb_buffer_size = %s\n", CB_BUFFER_SIZE);
        printf("ROMIO hint set: cb_nodes = %s\n", CB_NODES);
        printf("ROMIO hint set: striping_unit = %s\n", STRIPING_UNIT);
        printf("ROMIO hint set: striping_factor = %s\n", STRIPING_FACTOR);
    }

    // set hints
    err = MPI_Info_create (&info); MPI_CHECK(err);
    err = MPI_Info_set (info, "cb_buffer_size", CB_BUFFER_SIZE); MPI_CHECK (err);
    err = MPI_Info_set (info, "cb_nodes", CB_NODES); MPI_CHECK (err);
    err = MPI_Info_set (info, "striping_unit", STRIPING_UNIT); MPI_CHECK (err);
    err = MPI_Info_set (info, "striping_factor", STRIPING_FACTOR); MPI_CHECK (err);

    /* init buf */
    buf = (int*) malloc(sizeof(int) * len);
    for (i=0; i<len; i++)
        buf[i] = i + rank * len * len;

    /* create file */
    mode = MPI_MODE_CREATE | MPI_MODE_WRONLY;
    err = MPI_File_open(MPI_COMM_WORLD, filename, mode, info, &fh);
    MPI_CHECK(err);

    /* set atomicity to 1 */
    err = PMPI_File_set_atomicity(fh, 1);
    MPI_CHECK(err);

    /* write to file collectively */
    offset = rank * len * sizeof(int);
    err = MPI_File_write_at_all(fh, offset, buf, len, MPI_INT, MPI_STATUS_IGNORE);
    MPI_CHECK(err);

    err = MPI_File_close(&fh);
    MPI_CHECK(err);

    /* open file */
    mode = MPI_MODE_RDONLY;
    err = MPI_File_open(MPI_COMM_WORLD, filename, mode, info, &fh);
    MPI_CHECK(err);

    /* set atomicity to 1 */
    err = PMPI_File_set_atomicity(fh, 1);
    MPI_CHECK(err);

    /* read from file collectively */
    offset = rank * len * sizeof(int);
    err = MPI_File_read_at_all(fh, offset, buf, len, MPI_INT, MPI_STATUS_IGNORE);
    MPI_CHECK(err);

    err = MPI_File_close(&fh);
    MPI_CHECK(err);

    err = MPI_Info_free(&info);
    MPI_CHECK(err);

    free(buf);

err_out:
    MPI_Finalize();
    return (nerrs > 0);
}

