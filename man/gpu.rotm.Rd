\name{gpu.rotm}
\alias{gpu.rotm}
\title{Apply the modified Givens transformation}

\description{
	The function gpu.rotm applies the modified Givens transformation to the
	vectors x and y storing the resulting rotated vectors back into x and y 
	in gpu memory. Specifically, the 2 x min(length of x, length of y) matrix 
	where the first row is x and the second row is y is multiplied by the 2 x 2 
	h matrix specified in the gpu.vector sparam. As usual, the increment
	arguments along with the function advance.pointer can be used to produce 
	various exotic effects. See the CUBLAS library documentation from Nvidia
	for more information on the format for the sparam vector and the 
	increment effects.
}

\usage{
gpu.rotm(gpu.sparam, gpu.vector.x, gpu.vector.y, x.increment=1L,
	y.increment=1L)
}

\arguments{
	\item{sparam}{a 5 element gpu.vector consisting of first a flag and then of
		elements of the h matrix. See the CUBLAS library documentation for
		cublasDrotm for the complete specification.}
	\item{gpu.vector.x}{a list containing length and device.pointer. This is an
		object returned by gpu.set.vector in which the device.pointer addresses
		gpu memory. The entries of x get overwritten with the rotated vector x}
	\item{gpu.vector.y}{a gpu.vector (see gpu.vector.x).}
	\item{x.increment}{an integer determining the spacing between entries of
		x. If you want every other entry: increment=2L, every third: 
		increment=3L, and so on.}
	\item{y.increment}{an integer determining the spacing between entries of y.
		The entries of y get overwritten with the rotated vector y}
}

\value{
This function does not return an object to the R environment. The results are
written to gpu.vector.x and gpu.vector.y in gpu memory. If you are finished
with your linear algebra, you may retrieve the results using gpu.get.vector.
}

\seealso{
gpu.set.vector, gpu.get.vector, advance.pointer
}

\examples{
x <- runif(10)
gpu.x <- gpu.set.vector(x)

y <- runif(10)
gpu.y <- gpu.set.vector(y)

gpu.rot(2.0, 3.0, gpu.x, gpu.y)

new.x <- gpu.get.vector(gpu.x)
new.y <- gpu.get.vector(gpu.y)

}
