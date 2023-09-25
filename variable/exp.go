package variable

import "github.com/itsubaki/autograd/vector"

func Exp(x ...*Variable) *Variable {
	return (&Function{Forwarder: &ExpT{}}).ApplyAndFirst(x...)
}

type ExpT struct {
	y *Variable
}

func (f *ExpT) Forward(x ...*Variable) []*Variable {
	y := vector.Exp(x[0].Data)

	f.y = New(y...)
	return []*Variable{
		f.y,
	}
}

func (f *ExpT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Mul(gy[0], f.y), // gy * y
	}
}
