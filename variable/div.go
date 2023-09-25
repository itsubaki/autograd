package variable

import "github.com/itsubaki/autograd/vector"

func Div(x ...*Variable) *Variable {
	return (&Function{Forwarder: &DivT{}}).ApplyAndFirst(x...)
}

type DivT struct {
	x0, x1 *Variable
}

func (f *DivT) Forward(x ...*Variable) []*Variable {
	f.x0, f.x1 = x[0], x[1]

	y := vector.Div(f.x0.Data, f.x1.Data)
	return []*Variable{
		New(y...),
	}
}

func (f *DivT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Div(gy[0], f.x1),
		Mul(gy[0], Div(Neg(f.x0), Mul(f.x1, f.x1))), // gy * (-x0 / x1^2)
	}
}
