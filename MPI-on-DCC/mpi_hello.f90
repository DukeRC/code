program mpi_hello
  use mpi
  implicit none

  integer :: ierr, rank, nprocs, name_len
  character(len=MPI_MAX_PROCESSOR_NAME) :: hostname

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
  call MPI_Get_processor_name(hostname, name_len, ierr)

  write (*,'(A,I0,A,I0,A,A)') 'Hello from rank ', rank, ' of ', nprocs, &
                              ' on ', trim(hostname)

  call MPI_Finalize(ierr)
end program mpi_hello
