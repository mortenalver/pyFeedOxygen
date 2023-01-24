from mpi4py import MPI
import numpy as np

def mpiSettings():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    return comm, rank, size


def sendStateVector(comm, x, tag, dest):
    #print("send: "+str(x.shape)+", dest="+str(dest)+", tag="+str(tag))
    comm.Send([x, MPI.FLOAT], dest=dest, tag=tag)

def recvStateVector(comm, shp, tag, source):
    data = np.zeros(shp)
    #print("Receiving: "+str(data.shape)+", source="+str(source)+", tag="+str(tag))
    comm.Recv([data, MPI.FLOAT], source=source, tag=tag)
    return data