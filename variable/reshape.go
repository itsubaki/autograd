package variable

import "github.com/itsubaki/autograd/tensor"

// Reshape returns a function that reshapes x[0].
func Reshape(shape ...int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &ReshapeT{
			Shape: shape,
		},
	}).First
}

// ReshapeT is the differentiable reshape operation.
type ReshapeT struct {
	Shape  []int
	xShape []int
}

func (f *ReshapeT) Forward(x ...*Variable) []*Variable {
	f.xShape = x[0].Shape()

	y := tensor.Reshape(x[0].Data, f.Shape...)
	return []*Variable{
		From(y),
	}
}

func (f *ReshapeT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Reshape(f.xShape...)(gy[0]),
	}
}
