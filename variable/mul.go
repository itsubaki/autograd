package variable

import "github.com/itsubaki/autograd/vector"

func MulC(c float64, x ...*Variable) *Variable {
	return (&Function{Forwarder: &MulT{}}).ApplyAndFirst(Const(c), x[0])
}

func Mul(x ...*Variable) *Variable {
	return (&Function{Forwarder: &MulT{}}).ApplyAndFirst(x...)
}

type MulT struct {
	x0, x1 *Variable
}

func (f *MulT) Forward(x ...*Variable) []*Variable {
	f.x0, f.x1 = x[0], x[1]

	y := vector.Mul(f.x0.Data, f.x1.Data)
	return []*Variable{
		New(y...),
	}
}

func (f *MulT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Mul(gy[0], f.x1), // gy * x1
		Mul(gy[0], f.x0), // gy * x0
	}
}
