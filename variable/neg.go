package variable

import "github.com/itsubaki/autograd/tensor"

func Neg(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &NegT{},
	}).First(x...)
}

type NegT struct{}

func (f *NegT) Forward(x ...*Variable) []*Variable {
	return []*Variable{
		NewFrom(tensor.MulC(-1.0, x[0].Data)),
	}
}

func (f *NegT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Neg(gy[0]),
	}
}
