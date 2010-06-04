\name{gpu.get.vector}
\alias{gpu.get.vector}
\title{Copy a vector from gpu memory to host memory}

\description{
	The gpu.get.vector function copies a vector from gpu memory to host
	memory. The function returns a numeric R vector whose contents represent
	the contents of the gpu memory pointed to by the argument.
	
	Call this function only when you're sure that you're finished manipulating
	a vector. Gpu memory is high bandwith but very high latency!
}

\usage{
gpu.get.vector(gpu.vector, increment=1L)
}

\arguments{
	\item{gpu.vector}{
		a list containing a length and a device.pointer. The device.pointer
		element is a pointer addressing gpu memory. It should point to the
		first element of the vector that you want to retrieve.
	}
	\item{increment}{ 
		is a non-negative integer that defines the spacing of the vector
		elements within gpu memory. For example, if you wanted to retrieve
		only vector elements with odd indices, you could set increment to
		2L. If you wanted only even indexes, you could use the advance.pointer
		function and increment=2L. 
	}
}

\value{
The function returns a numeric R vector representing the contents of gpu memory.
}

\seealso{
advance.pointer, gpu.set.vector, is.gpu.vector, is.regular.vector, is.increment
}

\examples{
gpu.vector <- gpu.set.vector(runif(10))
regular.vector <- gpu.get.vector(gpu.vector)
}
