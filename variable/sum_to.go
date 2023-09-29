package variable

import "github.com/itsubaki/autograd/vector"

func SumTo(shape ...int) func(x ...*Variable) *Variable {
	return (&Function{Forwarder: &SumToT{Shape: shape}}).ApplyAndFirst
}

type SumToT struct {
	Shape, XShape []int
}

func (f *SumToT) Forward(x ...*Variable) []*Variable {
	f.XShape = vector.Shape(x[0].Data)

	y := vector.SumTo(f.Shape, x[0].Data)
	return []*Variable{
		New(y),
	}
}

func (f *SumToT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		BroadcastTo(f.XShape...)(gy[0]),
	}
}
