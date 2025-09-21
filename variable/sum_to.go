package variable

import "github.com/itsubaki/autograd/tensor"

func SumTo(shape ...int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &SumToT{
			Shape: shape,
		},
	}).First
}

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
