package variable

import "github.com/itsubaki/autograd/matrix"

func Cos(x ...*Variable) *Variable {
	return (&Function{Forwarder: &CosT{}}).First(x...)
}

type CosT struct {
	x *Variable
}

func (f *CosT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]
	y := matrix.Cos(x[0].Data)

	return []*Variable{
		NewFrom(y),
	}
}

func (f *CosT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Mul(Neg(Sin(f.x)), gy[0]), // -1.0 * sin(x) * gy
	}
}
