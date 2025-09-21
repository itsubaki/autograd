package variable

import "github.com/itsubaki/autograd/tensor"

func Sin(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &SinT{},
	}).First(x...)
}

type SinT struct {
	x *Variable
}

func (f *SinT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]

	y := tensor.Sin(x[0].Data)
	return []*Variable{
		From(y),
	}
}

func (f *SinT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Mul(Cos(f.x), gy[0]), // cos(x) * gy
	}
}
