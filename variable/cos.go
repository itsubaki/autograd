package variable

import (
	"github.com/itsubaki/autograd/vector"
)

func Cos(x ...*Variable) *Variable {
	return (&Function{Forwarder: &CosT{}}).ApplyAndFirst(x...)
}

type CosT struct {
	x *Variable
}

func (f *CosT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]

	y := vector.Cos(f.x.Data)
	return []*Variable{
		New(y...),
	}
}

func (f *CosT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Mul(Neg(Sin(f.x)), gy[0]), // -1.0 * sin(x) * gy
	}
}
