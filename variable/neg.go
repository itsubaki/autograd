package variable

import "github.com/itsubaki/autograd/matrix"

func Neg(x ...*Variable) *Variable {
	return (&Function{Forwarder: &NegT{}}).ApplyAndFirst(x...)
}

type NegT struct{}

func (f *NegT) Forward(x ...*Variable) []*Variable {
	y := matrix.MulC(-1.0, x[0].Data)
	return []*Variable{
		NewOf(y...),
	}
}

func (f *NegT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Neg(gy[0]),
	}
}
