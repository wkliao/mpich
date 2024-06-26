/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* tests noncontiguous reads/writes using collective I/O */

static void handle_error(int errcode, const char *str)
{
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errcode, msg, &resultlen);
    fprintf(stderr, "%s: %s\n", str, msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

#define MPI_CHECK(fn) { int errcode; errcode = (fn); if (errcode != MPI_SUCCESS) handle_error(errcode, #fn); }


#define SIZE 5000

#define VERBOSE 0
int main(int argc, char **argv)
{
    int *buf, i, mynod, nprocs, len;
    int err, errs = 0, toterrs;
    MPI_Aint disp, extent;
    MPI_File fh;
    MPI_Status status;
    char *filename;
    MPI_Datatype typevec, newtype, tmptype;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynod);

    if (nprocs != 2) {
        fprintf(stderr, "Run this program on two processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

/* process 0 takes the file name as a command-line argument and
   broadcasts it to other processes */
    if (!mynod) {
        i = 1;
        while ((i < argc) && strcmp("-fname", *argv)) {
            i++;
            argv++;
        }
        if (i >= argc) {
            fprintf(stderr, "\n*#  Usage: noncontig_coll -fname filename\n\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        argv++;
        len = strlen(*argv);
        filename = (char *) malloc(len + 1);
        strcpy(filename, *argv);
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(filename, len + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        filename = (char *) malloc(len + 1);
        MPI_Bcast(filename, len + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    buf = (int *) malloc(SIZE * sizeof(int));

    MPI_Type_vector(SIZE / 2, 1, 2, MPI_INT, &typevec);

    extent = SIZE * sizeof(int);

    len = 1;
    disp = mynod * sizeof(int);

    /* keep the struct, ditch the vector */
    MPI_Type_create_struct(1, &len, &disp, &typevec, &tmptype);
    MPI_Type_free(&typevec);

    MPI_Type_create_resized(tmptype, 0, extent, &newtype);
    MPI_Type_free(&tmptype);
    MPI_Type_commit(&newtype);

    if (!mynod) {
#if VERBOSE
        fprintf(stderr,
                "\ntesting noncontiguous in memory, noncontiguous in file using collective I/O\n");
#endif
        err = MPI_File_delete(filename, MPI_INFO_NULL);
        if (err != MPI_SUCCESS) {
            int errorclass;
            MPI_Error_class(err, &errorclass);
            if (errorclass != MPI_ERR_NO_SUCH_FILE)     /* ignore this error class */
                MPI_CHECK(err);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_CHECK(MPI_File_open(MPI_COMM_WORLD, filename,
                            MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh));

    MPI_CHECK(MPI_File_set_view(fh, 0, MPI_INT, newtype, "native", MPI_INFO_NULL));

    for (i = 0; i < SIZE; i++)
        buf[i] = i + mynod * SIZE;
    MPI_CHECK(MPI_File_write_all(fh, buf, 1, newtype, &status));

    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0; i < SIZE; i++)
        buf[i] = -1;

    MPI_CHECK(MPI_File_read_at_all(fh, 0, buf, 1, newtype, &status));

    for (i = 0; i < SIZE; i++) {
        if (!mynod) {
            if ((i % 2) && (buf[i] != -1)) {
                errs++;
                fprintf(stderr, "Process %d: buf %d is %d, should be -1\n", mynod, i, buf[i]);
            }
            if (!(i % 2) && (buf[i] != i)) {
                errs++;
                fprintf(stderr, "Process %d: buf %d is %d, should be %d\n", mynod, i, buf[i], i);
            }
        } else {
            if ((i % 2) && (buf[i] != i + mynod * SIZE)) {
                errs++;
                fprintf(stderr, "Process %d: buf %d is %d, should be %d\n",
                        mynod, i, buf[i], i + mynod * SIZE);
            }
            if (!(i % 2) && (buf[i] != -1)) {
                errs++;
                fprintf(stderr, "Process %d: buf %d is %d, should be -1\n", mynod, i, buf[i]);
            }
        }
    }

    MPI_CHECK(MPI_File_close(&fh));

    MPI_Barrier(MPI_COMM_WORLD);

    if (!mynod) {
#if VERBOSE
        fprintf(stderr,
                "\ntesting noncontiguous in memory, contiguous in file using collective I/O\n");
#endif
        err = MPI_File_delete(filename, MPI_INFO_NULL);
        if (err != MPI_SUCCESS) {
            int errorclass;
            MPI_Error_class(err, &errorclass);
            if (errorclass != MPI_ERR_NO_SUCH_FILE)     /* ignore this error class */
                MPI_CHECK(err);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_CHECK(MPI_File_open(MPI_COMM_WORLD, filename,
                            MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh));

    for (i = 0; i < SIZE; i++)
        buf[i] = i + mynod * SIZE;
    MPI_CHECK(MPI_File_write_at_all(fh, mynod * (SIZE / 2) * sizeof(int),
                                    buf, 1, newtype, &status));

    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0; i < SIZE; i++)
        buf[i] = -1;

    MPI_CHECK(MPI_File_read_at_all(fh, mynod * (SIZE / 2) * sizeof(int), buf, 1, newtype, &status));

    for (i = 0; i < SIZE; i++) {
        if (!mynod) {
            if ((i % 2) && (buf[i] != -1)) {
                errs++;
                fprintf(stderr, "Process %d: buf %d is %d, should be -1\n", mynod, i, buf[i]);
            }
            if (!(i % 2) && (buf[i] != i)) {
                errs++;
                fprintf(stderr, "Process %d: buf %d is %d, should be %d\n", mynod, i, buf[i], i);
            }
        } else {
            if ((i % 2) && (buf[i] != i + mynod * SIZE)) {
                errs++;
                fprintf(stderr, "Process %d: buf %d is %d, should be %d\n",
                        mynod, i, buf[i], i + mynod * SIZE);
            }
            if (!(i % 2) && (buf[i] != -1)) {
                errs++;
                fprintf(stderr, "Process %d: buf %d is %d, should be -1\n", mynod, i, buf[i]);
            }
        }
    }

    MPI_CHECK(MPI_File_close(&fh));

    MPI_Barrier(MPI_COMM_WORLD);

    if (!mynod) {
#if VERBOSE
        fprintf(stderr,
                "\ntesting contiguous in memory, noncontiguous in file using collective I/O\n");
#endif
        err = MPI_File_delete(filename, MPI_INFO_NULL);
        if (err != MPI_SUCCESS) {
            int errorclass;
            MPI_Error_class(err, &errorclass);
            if (errorclass != MPI_ERR_NO_SUCH_FILE)     /* ignore this error class */
                MPI_CHECK(err);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_CHECK(MPI_File_open(MPI_COMM_WORLD, filename,
                            MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh));

    MPI_CHECK(MPI_File_set_view(fh, 0, MPI_INT, newtype, "native", MPI_INFO_NULL));

    for (i = 0; i < SIZE; i++)
        buf[i] = i + mynod * SIZE;
    MPI_CHECK(MPI_File_write_all(fh, buf, SIZE, MPI_INT, &status));

    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0; i < SIZE; i++)
        buf[i] = -1;

    MPI_CHECK(MPI_File_read_at_all(fh, 0, buf, SIZE, MPI_INT, &status));

    for (i = 0; i < SIZE; i++) {
        if (!mynod) {
            if (buf[i] != i) {
                errs++;
                fprintf(stderr, "Process %d: buf %d is %d, should be %d\n", mynod, i, buf[i], i);
            }
        } else {
            if (buf[i] != i + mynod * SIZE) {
                errs++;
                fprintf(stderr, "Process %d: buf %d is %d, should be %d\n",
                        mynod, i, buf[i], i + mynod * SIZE);
            }
        }
    }

    MPI_CHECK(MPI_File_close(&fh));

    /* delete the file */
    if (!mynod)
        MPI_File_delete(filename, MPI_INFO_NULL);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allreduce(&errs, &toterrs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (mynod == 0) {
        if (toterrs > 0) {
            fprintf(stderr, "Found %d errors\n", toterrs);
        } else {
            fprintf(stdout, " No Errors\n");
        }
    }

    MPI_Type_free(&newtype);
    free(buf);
    free(filename);
    MPI_Finalize();
    return (toterrs > 0);
}
