package variable

import "github.com/itsubaki/autograd/matrix"

func Transpose(x ...*Variable) *Variable {
	return (&Function{Forwarder: &TransposeT{}}).ApplyAndFirst(x...)
}

type TransposeT struct{}

func (f *TransposeT) Forward(x ...*Variable) []*Variable {
	y := matrix.Transpose(x[0].Data)
	return []*Variable{
		NewOf(y...),
	}
}

func (f *TransposeT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Transpose(gy[0]),
	}
}
