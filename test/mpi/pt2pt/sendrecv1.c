/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include "mpitest.h"
#include "dtpools.h"
#include <assert.h>

/*
static char MTEST_Descrip[] = "Send-Recv";
*/


int main(int argc, char *argv[])
{
    int errs = 0, err;
    int rank, size, source, dest;
    int minsize = 2;
    int i, j, len, seed, testsize;
    MPI_Aint sendcount, recvcount, count[2];
    MPI_Aint maxbufsize;
    MPI_Comm comm;
    MPI_Datatype sendtype, recvtype;
    DTP_pool_s dtp;
    DTP_obj_s send_obj, recv_obj;
    void *sendbuf, *recvbuf;
    void *sendbuf_h, *recvbuf_h;
    char *basic_type;
    mtest_mem_type_e sendmem;
    mtest_mem_type_e recvmem;

    MTest_Init(&argc, &argv);

    MTestArgList *head = MTestArgListCreate(argc, argv);
    seed = MTestArgListGetInt(head, "seed");
    testsize = MTestArgListGetInt(head, "testsize");
    count[0] = MTestArgListGetLong(head, "sendcnt");
    count[1] = MTestArgListGetLong(head, "recvcnt");
    basic_type = MTestArgListGetString(head, "type");
    sendmem = MTestArgListGetMemType(head, "sendmem");
    recvmem = MTestArgListGetMemType(head, "recvmem");

    maxbufsize = MTestDefaultMaxBufferSize();

    err = DTP_pool_create(basic_type, count[0], seed, &dtp);
    if (err != DTP_SUCCESS) {
        fprintf(stderr, "Error while creating send pool (%s,%ld)\n", basic_type, count[0]);
        fflush(stderr);
    }

    MTestArgListDestroy(head);

    /* The following illustrates the use of the routines to
     * run through a selection of communicators and datatypes.
     * Use subsets of these for tests that do not involve combinations
     * of communicators, datatypes, and counts of datatypes */
    while (MTestGetIntracommGeneral(&comm, minsize, 1)) {
        if (comm == MPI_COMM_NULL) {
            /* for NULL comms, make sure these processes create the
             * same number of objects, so the target knows what
             * datatype layout to check for */
            errs += MTEST_CREATE_AND_FREE_DTP_OBJS(dtp, maxbufsize, testsize);
            errs += MTEST_CREATE_AND_FREE_DTP_OBJS(dtp, maxbufsize, testsize);
            continue;
        }

        /* Determine the sender and receiver */
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        source = 0;
        dest = size - 1;

        /* To improve reporting of problems about operations, we
         * change the error handler to errors return */
        MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);

        for (i = 0; i < testsize; i++) {
            err = DTP_obj_create(dtp, &send_obj, maxbufsize);
            if (err != DTP_SUCCESS) {
                errs++;
                break;
            }

            err = DTP_obj_create(dtp, &recv_obj, maxbufsize);
            if (err != DTP_SUCCESS) {
                errs++;
                break;
            }

            if (rank == source) {
                MTestAlloc(send_obj.DTP_bufsize, sendmem, &sendbuf_h, &sendbuf);
                assert(sendbuf && sendbuf_h);

                err = DTP_obj_buf_init(send_obj, sendbuf_h, 0, 1, count[0]);
                if (err != DTP_SUCCESS) {
                    errs++;
                    break;
                }
                MTestCopyContent(sendbuf_h, sendbuf, send_obj.DTP_bufsize, sendmem);

                sendcount = send_obj.DTP_type_count;
                sendtype = send_obj.DTP_datatype;

                err =
                    MPI_Send(sendbuf + send_obj.DTP_buf_offset, sendcount, sendtype, dest, 0, comm);
                if (err) {
                    errs++;
                    if (errs < 10) {
                        MTestPrintError(err);
                    }
                }

                MTestFree(sendmem, sendbuf_h, sendbuf);
            } else if (rank == dest) {
                MTestAlloc(recv_obj.DTP_bufsize, recvmem, &recvbuf_h, &recvbuf);
                assert(recvbuf && recvbuf_h);

                err = DTP_obj_buf_init(recv_obj, recvbuf_h, -1, -1, count[0]);
                if (err != DTP_SUCCESS) {
                    errs++;
                    break;
                }
                MTestCopyContent(recvbuf_h, recvbuf, recv_obj.DTP_bufsize, recvmem);

                recvcount = recv_obj.DTP_type_count;
                recvtype = recv_obj.DTP_datatype;

                err =
                    MPI_Recv(recvbuf + recv_obj.DTP_buf_offset, recvcount, recvtype, source, 0,
                             comm, MPI_STATUS_IGNORE);
                if (err) {
                    errs++;
                    if (errs < 10) {
                        MTestPrintError(err);
                    }
                }

                MTestCopyContent(recvbuf, recvbuf_h, recv_obj.DTP_bufsize, recvmem);
                err = DTP_obj_buf_check(recv_obj, recvbuf, 0, 1, count[0]);
                if (err != DTP_SUCCESS) {
                    if (errs < 10) {
                        char *recv_desc, *send_desc;
                        DTP_obj_get_description(recv_obj, &recv_desc);
                        DTP_obj_get_description(send_obj, &send_desc);
                        fprintf(stderr,
                                "Data in target buffer did not match for destination datatype %s and source datatype %s, count = %ld\n",
                                recv_desc, send_desc, count[0]);
                        free(recv_desc);
                        free(send_desc);
                    }
                    errs++;
                }

                MTestFree(recvmem, recvbuf_h, recvbuf);
            }
            DTP_obj_free(recv_obj);
            DTP_obj_free(send_obj);
        }
        MTestFreeComm(&comm);
    }

    DTP_pool_free(dtp);

    MTest_Finalize(errs);
    return MTestReturnValue(errs);
}
