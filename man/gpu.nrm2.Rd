\name{gpu.nrm2}
\alias{gpu.nrm2}
\title{Calculate the Euclidean norm of a gpu.vector}

\description{
	The gpu.nrm2 function calculates the Euclidean norm of a vector.
	The vector resides in gpu memory and the norm is returned to the R
	environment (host memory). You can use the increment argument and
	the advance.pointer function to norm up a subvector.
}

\usage{
gpu.nrm2(gpu.vector.x, x.increment=1L)
}

\arguments{
	\item{gpu.vector.x}{a list containing length and device.pointer. This is an
		object returned by gpu.set.vector in which the device.pointer addresses
		gpu memory.}
	\item{x.increment}{an integer determining the spacing between vector
		entries. If you want every other entry: increment=2L, every third:
		increment=3L, and so on.}  
}

\value{
	a single numeric value representing the vector's norm
}

\seealso{
gpu.set.vector, advance.pointer
}

\examples{
x <- runif(10)
gpu.x <- gpu.set.vector(x)
x.norm <- gpu.nrm2(gpu.x)
}
