package function

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

// SoftmaxSimple applies softmax along the given axis using the simple helper API.
func SoftmaxSimple(x *variable.Variable, axis int) *variable.Variable {
	shape := tensor.KeepDims(x.Shape(), []int{axis})

	y := Exp(x)
	sumy := SumTo(shape...)(y)
	return Div(y, sumy)
}
