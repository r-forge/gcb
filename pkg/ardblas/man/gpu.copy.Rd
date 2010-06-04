\name{gpu.copy}
\alias{gpu.copy}
\title{On-gpu vector copy routine}

\description{
	The gpu.copy routine copies the contents of vector x into vector y
	overwriting the contents of y. You can vary the increments arguments and 
	use the advance.pointer function to achieve various effects (e.g. writing
	the odd entries of x to the even entries of y)...
}

\usage{
gpu.copy(gpu.vector.x, gpu.vector.y, x.increment=1L, y.increment=1L)
}

\arguments{
	\item{gpu.vector.x}{a list containing length and device.pointer. This is an
		object returned by gpu.set.vector in which the device.pointer addresses
		gpu memory.}
	\item{gpu.vector.y}{a gpu.vector (see gpu.vector.x). This vector is
		overwritten by x!}
	\item{x.increment}{an integer determining the spacing between entries of
		x. If you want every other entry: increment=2L, every third: 
		increment=3L, and so on.}
	\item{y.increment}{an integer determining the spacing between entries of y.}
}

\value{
This function does not return an object to the R environment. The entries of x
are written to y in gpu memory. If you are finished with your linear algebra,
you may retrieve the results using gpu.get.vector.
}

\seealso{
gpu.set.vector, gpu.get.vector, advance.pointer
}

\examples{
x <- runif(10)
gpu.x <- gpu.set.vector(x)
gpu.y <- gpu.set.vector(double(length(x)))
gpu.copy(gpu.x, gpu.y)
y <- gpu.get.vector(gpu.y)
}
