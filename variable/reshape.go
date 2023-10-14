package variable

import "github.com/itsubaki/autograd/matrix"

func Reshape(shape ...int) func(x ...*Variable) *Variable {
	return (&Function{Forwarder: &ReshapeT{Shape: shape}}).First
}

type ReshapeT struct {
	Shape, xShape []int
}

func (f *ReshapeT) Forward(x ...*Variable) []*Variable {
	f.xShape = Shape(x[0])

	y := matrix.Reshape(f.Shape, x[0].Data)
	return []*Variable{
		NewOf(y...),
	}
}

func (f *ReshapeT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Reshape(f.xShape...)(gy[0]),
	}
}
