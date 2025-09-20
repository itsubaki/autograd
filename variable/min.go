package variable

import "github.com/itsubaki/autograd/tensor"

func Min(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &MinT{},
	}).First(x...)
}

type MinT struct {
	MaxT
}

func (f *MinT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]
	f.y = NewFrom(tensor.Min(x[0].Data))

	return []*Variable{
		f.y,
	}
}
