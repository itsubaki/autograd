package variable

import "github.com/itsubaki/autograd/tensor"

func Reshape(shape ...int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &ReshapeT{
			Shape: shape,
		},
	}).First
}

type ReshapeT struct {
	Shape  []int
	xShape []int
}

func (f *ReshapeT) Forward(x ...*Variable) []*Variable {
	f.xShape = x[0].Shape()

	return []*Variable{
		NewFrom(tensor.Reshape(x[0].Data, f.Shape...)),
	}
}

func (f *ReshapeT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Reshape(f.xShape...)(gy[0]),
	}
}
