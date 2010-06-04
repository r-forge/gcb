\name{gpu.swap}
\alias{gpu.swap}
\title{Swap entries between gpu.vectors}

\description{
	The gpu.swap function swaps the entries of the vectors x and y. The original
	entries of x and y are overwritten. You can vary the increments arguments and 
	use the advance.pointer function to achieve various effects (e.g. swapping
	the odd entries of x to with the even entries of y)...
}

\usage{
gpu.swap(gpu.vector.x, gpu.vector.y, x.increment=1L, y.increment=1L)
}

\arguments{
	\item{gpu.vector.x}{a list containing length and device.pointer. This is an
		object returned by gpu.set.vector in which the device.pointer addresses
		gpu memory.}
	\item{gpu.vector.y}{a gpu.vector (see gpu.vector.x).}
	\item{x.increment}{an integer determining the spacing between entries of
		x. If you want every other entry: increment=2L, every third: 
		increment=3L, and so on.}
	\item{y.increment}{an integer determining the spacing between entries of y.}
}

\value{
This function does not return an object to the R environment. The result is
written to gpu.vector.y and gpu.vector.x in gpu memory. If you are finished
with your linear algebra, you may retrieve the results using gpu.get.vector.
}

\seealso{
gpu.set.vector, gpu.get.vector, advance.pointer
}

\examples{
# swap the even entries of x with the odd entries
x <- runif(10)
gpu.x <- gpu.set.vector(x)

gpu.odds <- gpu.x
gpu.odds$length <- 5L

gpu.evens <- gpu.x
gpu.evens$length <- 5L
gpu.evens$device.pointer <- advance.pointer(gpu.x$device.pointer, 1L)

gpu.swap(gpu.odds, gpu.evens, 2L, 2L)
mixed.x <- gpu.get.vector(gpu.x)
}
