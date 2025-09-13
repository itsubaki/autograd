package variable

import "github.com/itsubaki/autograd/tensor"

func Transpose(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &TransposeT{},
	}).First(x...)
}

type TransposeT struct{}

func (f *TransposeT) Forward(x ...*Variable) []*Variable {
	y := tensor.Transpose(x[0].Data)
	return []*Variable{
		NewFrom(y),
	}
}

func (f *TransposeT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Transpose(gy[0]),
	}
}
