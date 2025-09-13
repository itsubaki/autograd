package variable

import (
	"github.com/itsubaki/autograd/tensor"
)

func SumTo(axes ...int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &SumToT{
			Axes: axes,
		},
	}).First
}

type SumToT struct {
	Axes   []int
	xShape []int
}

func (f *SumToT) Forward(x ...*Variable) []*Variable {
	f.xShape = x[0].Shape()
	y := tensor.Sum(x[0].Data, f.Axes...)

	return []*Variable{
		NewFrom(y),
	}
}

func (f *SumToT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		BroadcastTo(f.xShape...)(gy[0]),
	}
}
