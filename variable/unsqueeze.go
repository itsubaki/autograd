package variable

import "github.com/itsubaki/autograd/tensor"

func Unsqueeze(axis int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &UnsqueezeT{
			Axis: axis,
		},
	}).First
}

type UnsqueezeT struct {
	Axis int
}

func (f *UnsqueezeT) Forward(x ...*Variable) []*Variable {
	return []*Variable{
		From(tensor.Unsqueeze(x[0].Data, f.Axis)),
	}
}

func (f *UnsqueezeT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Squeeze(f.Axis)(gy...),
	}
}
