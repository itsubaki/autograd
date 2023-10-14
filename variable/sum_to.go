package variable

import "github.com/itsubaki/autograd/matrix"

func SumTo(shape ...int) func(x ...*Variable) *Variable {
	return (&Function{Forwarder: &SumToT{Shape: shape}}).First
}

type SumToT struct {
	Shape, xShape []int
}

func (f *SumToT) Forward(x ...*Variable) []*Variable {
	f.xShape = Shape(x[0])

	y := matrix.SumTo(f.Shape, x[0].Data)
	return []*Variable{
		NewOf(y...),
	}
}

func (f *SumToT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		BroadcastTo(f.xShape...)(gy[0]),
	}
}
