package variable

import "github.com/itsubaki/autograd/tensor"

// SumTo returns a function that reduces x[0] to the given shape.
func SumTo(shape ...int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &SumToT{
			Shape: shape,
		},
	}).First
}

// SumToT is the differentiable SumTo operation.
type SumToT struct {
	Shape  []int
	xShape []int
}

func (f *SumToT) Forward(x ...*Variable) []*Variable {
	f.xShape = x[0].Shape()

	y := tensor.SumTo(x[0].Data, f.Shape...)
	return []*Variable{
		From(y),
	}
}

func (f *SumToT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		BroadcastTo(f.xShape...)(gy[0]),
	}
}
