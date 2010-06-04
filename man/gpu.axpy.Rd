\name{gpu.axpy}
\alias{gpu.axpy}
\title{On-gpu axpy routine}

\description{
	The gpu.axpy function does the following: y <- alpha * x + y where alpha is
	a scalar and x and y are gpu.vectors (see gpu.set.vector). Calculations and
	data reside (and stay) on the gpu. The vector y is overwritten with the
	result.
}

\usage{
gpu.axpy(gpu.vector.x, gpu.vector.y, alpha=1.0, x.increment = 1L,
	y.increment = 1L)
}

\arguments{
	\item{gpu.vector.x}{a list containing length and device.pointer. This is an
		object returned by gpu.set.vector in which the device.pointer addresses
		gpu memory.}
	\item{gpu.vector.y}{a gpu.vector (see gpu.vector.x). This vector is
		overwritten by the result!}
	\item{alpha}{a single numeric value -- the scalar coefficient for vector x.}
	\item{x.increment}{an integer determining the spacing between entries of
		x. If you want every other entry: increment=2L, every third: 
		increment=3L, and so on.}
	\item{y.increment}{an integer determining the spacing between entries of
		y.}
}

\value{
This function does not return an object to the R environment. The result is
written to gpu.vector.y in gpu memory. If you are finished with your linear
algebra, you may retrieve the results using gpu.get.vector.
}

\seealso{
gpu.set.vector, gpu.get.vector, advance.pointer
}

\examples{
x <- runif(10)
gpu.x <- gpu.set.vector(x)

y <- runif(10)
gpu.y <- gpu.set.vector(y)

gpu.axp(gpu.x, gpu.y, 5.0)
z <- gpu.get.vector(gpu.y)
}
