package variable

import "github.com/itsubaki/autograd/tensor"

// BroadcastTo returns a function that broadcasts x[0] to the given shape.
func BroadcastTo(shape ...int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &BroadcastToT{
			Shape: shape,
		},
	}).First
}

// BroadcastToT is the differentiable BroadcastTo operation.
type BroadcastToT struct {
	Shape  []int
	xShape []int
}

func (f *BroadcastToT) Forward(x ...*Variable) []*Variable {
	f.xShape = x[0].Shape()

	y := tensor.BroadcastTo(x[0].Data, f.Shape...)
	return []*Variable{
		From(y),
	}
}

func (f *BroadcastToT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		SumTo(f.xShape...)(gy[0]),
	}
}
