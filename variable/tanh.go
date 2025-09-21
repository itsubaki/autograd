package variable

import "github.com/itsubaki/autograd/matrix"

func Tanh(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &TanhT{},
	}).First(x...)
}

type TanhT struct {
	y *Variable
}

func (f *TanhT) Forward(x ...*Variable) []*Variable {
	y := matrix.Tanh(x[0].Data)
	f.y = From(y)

	return []*Variable{
		f.y,
	}
}

func (f *TanhT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Mul(gy[0], SubC(1.0, Mul(f.y, f.y))), // gy * (1-y^2)
	}
}
