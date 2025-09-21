package variable

import "github.com/itsubaki/autograd/tensor"

func Transpose(axes ...int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &TransposeT{
			Axes: axes,
		},
	}).First
}

type TransposeT struct {
	Axes []int
}

func (f *TransposeT) Forward(x ...*Variable) []*Variable {
	y := tensor.Transpose(x[0].Data, f.Axes...)
	return []*Variable{
		From(y),
	}
}

func (f *TransposeT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Transpose(f.Axes...)(gy[0]),
	}
}
