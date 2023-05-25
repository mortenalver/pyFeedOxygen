from mpi4py import MPI
import numpy as np

padding = 2

def mpiSettings():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    return comm, rank, size

def getSplits(dim, size):
    splits = []
    paddedSplits = []
    iStart = 0
    iEnd = int(np.floor(dim/size))
    splits.append((iStart, iEnd))
    paddedSplits.append((iStart, iEnd+padding))
    for i in range(1,size-1):
        iStart = int(np.floor((i*dim / size)))
        iEnd = int(np.floor((i+1)*dim / size))
        splits.append((iStart, iEnd))
        paddedSplits.append((iStart-padding, iEnd+padding))
    iStart = int(np.floor(((size-1)*dim / size)))
    iEnd = dim
    splits.append((iStart, iEnd))
    paddedSplits.append((iStart-padding, iEnd))
    return splits, paddedSplits

# When this is called, all ranks > 0 should send their parts of the two fields to rank 0,
# and rank 0 should collect all of it.
def syncToRank0(comm, rank, N, cageDims, xBounds, fc, o2, lostFeed):
    if rank > 0:
        # Send o2 field:
        comm.Send([o2[xBounds[0]:xBounds[1],:,:], MPI.FLOAT], dest=0, tag=1)
        comm.Send([fc[xBounds[0]:xBounds[1], :, :], MPI.FLOAT], dest=0, tag=2)
        data = np.empty((1))
        data[0] = lostFeed
        comm.Send([data, MPI.FLOAT], dest=0, tag=3)
        return 0
    else:
        lostFeedTotal = 0.
        splits, paddedSplits = getSplits(cageDims[0], N)
        for i in range(1, N):
            xRange = splits[i]
            data = np.empty((xRange[1]-xRange[0], cageDims[1], cageDims[2]))
            comm.Recv([data, MPI.FLOAT], source=i, tag=1)
            o2[xRange[0]:xRange[1],:,:] = data[...]
            comm.Recv([data, MPI.FLOAT], source=i, tag=2)
            fc[xRange[0]:xRange[1], :, :] = data[...]
            data = np.empty((1,))
            comm.Recv([data, MPI.FLOAT], source=i, tag=3)
            lostFeedTotal += data[0]
        return lostFeedTotal

# When this is called, rank 0 should broadcast the entire fields to all other ranks:
def distAllFromRank0(comm, rank, N, cageDims, xBounds, fc, o2):
    comm.Bcast([o2, MPI.FLOAT], root=0)
    comm.Bcast([fc, MPI.FLOAT], root=0)

# When this is called, rank 0 should broadcast the appropriate padded portion of the fields
# to all other ranks:
def distFromRank0(comm, rank, N, cageDims, xBounds, fc, o2):
    splits, paddedSplits = getSplits(cageDims[0], N)
    if rank==0:
        for i in range(1, N):
            boundsNow = paddedSplits[i]
            comm.Send([o2[boundsNow[0]:boundsNow[1], :, :], MPI.FLOAT], dest=i, tag=1)
            comm.Send([fc[boundsNow[0]:boundsNow[1], :, :], MPI.FLOAT], dest=i, tag=2)
    else:
        boundsNow = paddedSplits[rank]
        data = np.empty((boundsNow[1]-boundsNow[0], cageDims[1], cageDims[2]))
        comm.Recv([data, MPI.FLOAT], source=0, tag=1)
        o2[boundsNow[0]:boundsNow[1], :, :] = data[...]
        comm.Recv([data, MPI.FLOAT], source=0, tag=2)
        fc[boundsNow[0]:boundsNow[1], :, :] = data[...]

