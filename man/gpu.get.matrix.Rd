\name{gpu.get.matrix}
\alias{gpu.get.matrix}
\title{Copy a matrix from gpu memory to host memory}

\description{
	The gpu.get.matrix function copies a matrix from gpu memory to host
	memory. The function returns a numeric R matrix whose contents represent
	the contents of the gpu memory pointed to by the argument.
	
	Call this function only when you're sure that you're finished manipulating
	a matrix. Gpu memory is high bandwith but very high latency!
}

\usage{
gpu.get.matrix(gpu.matrix, matrix.ld = NULL)
}

\arguments{
	\item{gpu.matrix}{
		a list containing a length and a device.pointer. The device.pointer
		element is a pointer addressing gpu memory. It should point to the
		first element of the matrix that you want to retrieve.
	}
	\item{matrix.ld}{ 
	The matrix.ld defines the space needed to get to the
	next row in matrix element units. For example, an leading.dimension of 5L
	means 'the next row is 5 elements from the beginning of the current row'.
	You can set the matrix.ld to something larger than the number of rows to
	select a submatrix.
	}
}

\value{
The function returns a numeric R matrix representing the contents of gpu memory.
}

\seealso{
advance.pointer, gpu.set.matrix, is.gpu.matrix, is.regular.matrix,
is.leading.dimension
}

\examples{
gpu.matrix <- gpu.set.matrix(matrix(runif(10), 5, 2))
regular.matrix <- gpu.get.matrix(gpu.matrix)
}
