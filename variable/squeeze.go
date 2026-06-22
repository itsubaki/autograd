package variable

import "github.com/itsubaki/autograd/tensor"

func Squeeze(axis int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &SqueezeT{
			Axis: axis,
		},
	}).First
}

type SqueezeT struct {
	Axis int
}

func (f *SqueezeT) Forward(x ...*Variable) []*Variable {
	return []*Variable{
		From(tensor.Squeeze(x[0].Data, f.Axis)),
	}
}

func (f *SqueezeT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Unsqueeze(f.Axis)(gy...),
	}
}
