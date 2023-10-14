package variable

import "github.com/itsubaki/autograd/matrix"

func BroadcastTo(shape ...int) func(x ...*Variable) *Variable {
	return (&Function{Forwarder: &BroadcastToT{Shape: shape}}).First
}

type BroadcastToT struct {
	Shape, xShape []int
}

func (f *BroadcastToT) Forward(x ...*Variable) []*Variable {
	f.xShape = Shape(x[0])

	y := matrix.BroadcastTo(f.Shape, x[0].Data)
	return []*Variable{
		NewOf(y...),
	}
}

func (f *BroadcastToT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		SumTo(f.xShape...)(gy[0]),
	}
}
