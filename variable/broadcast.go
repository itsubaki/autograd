package variable

import (
	"github.com/itsubaki/autograd/vector"
)

func Broadcast(n int) func(x ...*Variable) *Variable {
	return (&Function{Forwarder: &BroadcastT{Shape: n}}).ApplyAndFirst
}

type BroadcastT struct {
	Shape int
}

func (f *BroadcastT) Forward(x ...*Variable) []*Variable {
	y, _ := vector.Broadcast(x[0].Data, make([]float64, f.Shape))
	return []*Variable{
		New(y...),
	}
}

func (f *BroadcastT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Sum(gy[0]),
	}
}
