\name{gpu.asum}
\alias{gpu.asum}
\title{Sum the absolute values of the entries of a gpu.vector}

\description{
	The gpu.asum function sums the absolute values of the entries of a vector.
	The vector resides in gpu memory and the sum is returned to the R
	environment (host memory). You can use the increment argument and
	the advance.pointer function to asum up a subvector.
}

\usage{
gpu.asum(gpu.vector, increment=1L)
}

\arguments{
	\item{gpu.vector}{a list containing length and device.pointer. This is an
		object returned by gpu.set.vector in which the device.pointer addresses
		gpu memory.}
	\item{increment}{an integer determining the spacing between vector entries.
		If you want every other entry: increment=2L, every third: 
		increment=3L, and so on.}  
}

\value{
	a single numeric value representing the sum of the absolute values of
	vector entries
}

\seealso{
gpu.get.vector, advance.pointer
}

\examples{
x <- runif(10)
gpu.x <- gpu.set.vector(x)
x.asum <- gpu.asum(gpu.x)
}
