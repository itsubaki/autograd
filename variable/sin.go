package variable

import "github.com/itsubaki/autograd/vector"

func Sin(x ...*Variable) *Variable {
	return (&Function{Forwarder: &SinT{}}).ApplyAndFirst(x...)
}

type SinT struct {
	x *Variable
}

func (f *SinT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]

	y := vector.Sin(f.x.Data)
	return []*Variable{
		New(y...),
	}
}

func (f *SinT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Mul(Cos(f.x), gy[0]), // cos(x) * gy
	}
}
