\name{gpu.dot}
\alias{gpu.dot}
\title{On-gpu vector dot product routine}

\description{
	The gpu.dot routine computes the dot product of x and y using the gpu 
	where x and y are stored in gpu memory. The result is returned to the
	R environment (host memory). You can vary the increments arguments and 
	use the advance.pointer function to achieve various effects (e.g. taking
	the dot product of the odd entries of x to with the even entries of y)...
}

\usage{
gpu.dot(gpu.vector.x, gpu.vector.y, x.increment=1L, y.increment=1L)
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
	a single numeric value representing the dot product of the vectors x and y
}

\seealso{
gpu.set.vector, advance.pointer
}

\examples{
x <- runif(10)
gpu.x <- gpu.set.vector(x)

y <- runif(10)
gpu.y <- gpu.set.vector(y)

x.dot.y <- gpu.dot(gpu.x, gpu.y)
}
