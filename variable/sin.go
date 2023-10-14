package variable

import "github.com/itsubaki/autograd/matrix"

func Sin(x ...*Variable) *Variable {
	return (&Function{Forwarder: &SinT{}}).First(x...)
}

type SinT struct {
	x *Variable
}

func (f *SinT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]

	y := matrix.Sin(x[0].Data)
	return []*Variable{
		NewOf(y...),
	}
}

func (f *SinT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Mul(Cos(f.x), gy[0]), // cos(x) * gy
	}
}
