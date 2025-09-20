package variable

import "github.com/itsubaki/autograd/tensor"

func BroadcastTo(shape ...int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &BroadcastToT{
			Shape: shape,
		},
	}).First
}

type BroadcastToT struct {
	Shape, xShape []int
}

func (f *BroadcastToT) Forward(x ...*Variable) []*Variable {
	f.xShape = x[0].Shape()

	return []*Variable{
		NewFrom(tensor.BroadcastTo(x[0].Data, f.Shape...)),
	}
}

func (f *BroadcastToT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		SumTo(f.xShape...)(gy[0]),
	}
}
