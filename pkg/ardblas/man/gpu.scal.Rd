\name{gpu.scal}
\alias{gpu.scal}
\title{Scale a gpu.vector}

\description{
	The gpu.scal function scales the entries of a vector x in gpu memory by a 
	factor of alpha. So the ith entry xi of x is overwritten with alpha * xi.
}

\usage{
gpu.scal(alpha, gpu.vector.x, x.increment=1L)
}

\arguments{
	\item{alpha}{a single numeric value used to scale the entries of x}
	\item{gpu.vector.x}{a list containing length and device.pointer. This is an
		object returned by gpu.set.vector in which the device.pointer addresses
		gpu memory.}
	\item{x.increment}{an integer determining the spacing between vector
		entries. If you want every other entry: increment=2L, every third:
		increment=3L, and so on.}  
}

\value{
This function does not return an object to the R environment. The result is
written to gpu.vector.x in gpu memory. If you are finished with your linear
algebra, you may retrieve the results using gpu.get.vector.
}

\seealso{
gpu.set.vector, gpu.get.vector, advance.pointer
}

\examples{
x <- runif(10)
gpu.x <- gpu.set.vector(x)
gpu.scal(5.0, gpu.x)
new.x <- gpu.get.vector(gpu.x)
}
