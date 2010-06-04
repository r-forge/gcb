\name{gpu.set.vector}
\alias{gpu.set.vector}
\title{Copy a vector to gpu memory}

\description{
	The gpu.set.vector function copies a vector to gpu memory. You may use the
	returned object as an argument to the ardblas library's linear algebra
	routines. When you're finished manipulating the vector, you may retrieve
	the entries of the vector by calling gpu.get.vector.
	
	This function is possibly very time consuming: Gpu memory is high bandwith
	but very high latency!
}

\usage{
gpu.set.vector(v)
}

\arguments{
	\item{v}{a non-NULL numeric vector that the user wishes to copy to gpu
		memory}  
}

\value{
	After copying the vector to gpu memory, the function returns a list
	containing length and device.pointer. The length and device.pointer elements
	may be manipulated by the user to achieve various effects. To err on the
	side of caution, please use the advance.pointer function to manipulate
	the device.pointer element. The list returned is ment to be used as an
	argument to the linear algebra routines. Once the user is finished with the
	vector, the user may get the possibly modified vector by calling
	gpu.get.vector.
}

\seealso{
gpu.get.vector, advance.pointer, is.regular.vector
}

\examples{
gpu.vector <- gpu.set.vector(runif(10))
regular.vector <- gpu.get.vector(gpu.vector)
}
