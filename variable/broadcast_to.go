package variable

import "github.com/itsubaki/autograd/vector"

func BroadcastTo(shape ...int) func(x ...*Variable) *Variable {
	return (&Function{Forwarder: &BroadcastToT{Shape: shape}}).ApplyAndFirst
}

type BroadcastToT struct {
	Shape, XShape []int
}

func (f *BroadcastToT) Forward(x ...*Variable) []*Variable {
	f.XShape = vector.Shape(x[0].Data)

	y := vector.BroadcastTo(f.Shape, x[0].Data)
	return []*Variable{
		New(y...),
	}
}

func (f *BroadcastToT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		SumTo(f.XShape...)(gy[0]),
	}
}