package variable

import "github.com/itsubaki/autograd/vector"

func SumTo(shape ...int) func(x ...*Variable) *Variable {
	return (&Function{Forwarder: &SumToT{Shape: shape}}).ApplyAndFirst
}

type SumToT struct {
	Shape, xShape []int
}

func (f *SumToT) Forward(x ...*Variable) []*Variable {
	f.xShape = vector.Shape(x[0].Data)

	y := vector.SumTo(f.Shape, x[0].Data)
	return []*Variable{
		New(y),
	}
}

func (f *SumToT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		BroadcastTo(f.xShape...)(gy[0]),
	}
}
