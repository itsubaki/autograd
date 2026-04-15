package function

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

// SoftmaxSimple applies the softmax function along the given axis
// by composing primitive operations such as Exp, SumTo, and Div.
func SoftmaxSimple(x *variable.Variable, axis int) *variable.Variable {
	shape := tensor.KeepDims(x.Shape(), []int{axis})

	y := Exp(x)
	sumy := SumTo(shape...)(y)
	return Div(y, sumy)
}
