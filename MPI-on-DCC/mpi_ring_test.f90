program mpi_ring_test
  use mpi
  implicit none

  integer :: ierr
  integer :: rank, nprocs, name_len
  integer :: left_rank, right_rank
  integer :: sendbuf, recvbuf, rank_sum
  character(len=MPI_MAX_PROCESSOR_NAME) :: hostname

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
  call MPI_Get_processor_name(hostname, name_len, ierr)

  write (*,'(A,I0,A,I0,A,A)') 'Rank ', rank, ' of ', nprocs, ' on ', trim(hostname)
  call MPI_Barrier(MPI_COMM_WORLD, ierr)

  left_rank = mod(rank - 1 + nprocs, nprocs)
  right_rank = mod(rank + 1, nprocs)
  sendbuf = rank
  recvbuf = -1

  ! Exchange one integer with neighboring ranks to verify point-to-point traffic.
  call MPI_Sendrecv(sendbuf, 1, MPI_INTEGER, right_rank, 0, &
                    recvbuf, 1, MPI_INTEGER, left_rank, 0, &
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)

  write (*,'(A,I0,A,I0,A,I0)') 'Rank ', rank, ' received ', recvbuf, ' from rank ', left_rank

  call MPI_Allreduce(rank, rank_sum, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
  if (rank == 0) then
    write (*,'(A,I0)') 'Global rank sum = ', rank_sum
  end if

  call MPI_Finalize(ierr)
end program mpi_ring_test
