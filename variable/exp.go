package variable

import "github.com/itsubaki/autograd/tensor"

// Exp applies the exponential function.
func Exp(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &ExpT{},
	}).First(x...)
}

// ExpT is the differentiable exponential operation.
type ExpT struct {
	y *Variable
}

func (f *ExpT) Forward(x ...*Variable) []*Variable {
	f.y = From(tensor.Exp(x[0].Data))
	return []*Variable{
		f.y,
	}
}

func (f *ExpT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Mul(gy[0], f.y), // gy * y
	}
}
