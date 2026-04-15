package variable

import "github.com/itsubaki/autograd/tensor"

// Cos applies the cosine function.
func Cos(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &CosT{},
	}).First(x...)
}

// CosT is the differentiable cosine operation.
type CosT struct {
	x *Variable
}

func (f *CosT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]

	y := tensor.Cos(x[0].Data)
	return []*Variable{
		From(y),
	}
}

func (f *CosT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Mul(Neg(Sin(f.x)), gy[0]), // -1.0 * sin(x) * gy
	}
}
