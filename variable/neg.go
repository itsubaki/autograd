package variable

import "github.com/itsubaki/autograd/vector"

func Neg(x ...*Variable) *Variable {
	return (&Function{Forwarder: &NegT{}}).ApplyAndFirst(x...)
}

type NegT struct{}

func (f *NegT) Forward(x ...*Variable) []*Variable {
	y := vector.MulC(-1.0, x[0].Data)
	return []*Variable{
		New(y...),
	}
}

func (f *NegT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Neg(gy[0]),
	}
}
