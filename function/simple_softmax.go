package function

import "github.com/itsubaki/autograd/variable"

func SoftmaxSimple(x *variable.Variable, axis int) *variable.Variable {
	shape := x.Shape()
	shape[axis] = 1

	y := Exp(x)
	sumy := SumTo(shape...)(y)
	return Div(y, sumy)
}
