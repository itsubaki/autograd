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

	return []*Variable{
		NewFrom(tensor.Sin(x[0].Data)),
	}
}

func (f *SinT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Mul(Cos(f.x), gy[0]), // cos(x) * gy
	}
}
