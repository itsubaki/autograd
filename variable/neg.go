package variable

import "github.com/itsubaki/autograd/tensor"

// Neg returns a variable representing -x[0].
func Neg(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &NegT{},
	}).First(x...)
}

// NegT is the differentiable negation operation.
type NegT struct{}

func (f *NegT) Forward(x ...*Variable) []*Variable {
	y := tensor.MulC(-1.0, x[0].Data)
	return []*Variable{
		From(y),
	}
}

func (f *NegT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Neg(gy[0]),
	}
}
