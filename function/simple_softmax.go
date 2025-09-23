package function

import "github.com/itsubaki/autograd/variable"

func SoftmaxSimple(x *variable.Variable, axis int) *variable.Variable {
	shape := keepDims(x.Shape(), axis)

	y := Exp(x)
	sumy := SumTo(shape...)(y)
	return Div(y, sumy)
}
