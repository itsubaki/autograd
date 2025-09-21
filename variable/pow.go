package variable

import "github.com/itsubaki/autograd/matrix"

func Pow(p float64) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &PowT{
			P: p,
		},
	}).First
}

type PowT struct {
	P float64
	x *Variable
}

func (f *PowT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]

	y := matrix.Pow(f.P, x[0].Data)
	return []*Variable{
		From(y),
	}
}

func (f *PowT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Mul(gy[0], MulC(f.P, Pow(f.P-1)(f.x))), // gy * c * x^(c-1)
	}
}
