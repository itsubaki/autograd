package variable

import "github.com/itsubaki/autograd/vector"

func Pow(c float64) func(x ...*Variable) *Variable {
	return (&Function{Forwarder: &PowT{C: c}}).ApplyAndFirst
}

type PowT struct {
	C float64
	x *Variable
}

func (f *PowT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]

	y := vector.Pow(f.C, x[0].Data)
	return []*Variable{
		New(y...),
	}
}

func (f *PowT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Mul(gy[0], MulC(f.C, Pow(f.C-1)(f.x))), // gy * c * x^(c-1)
	}
}
