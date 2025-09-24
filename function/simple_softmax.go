package function

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func SoftmaxSimple(x *variable.Variable, axis int) *variable.Variable {
	shape := tensor.KeepDims(x.Shape(), []int{axis})

	y := Exp(x)
	sumy := SumTo(shape...)(y)
	return Div(y, sumy)
}
