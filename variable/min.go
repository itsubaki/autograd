package variable

import "github.com/itsubaki/autograd/tensor"

func Min(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &MinT{},
	}).First(x...)
}

type MinT struct {
	x, y *Variable
}

func (f *MinT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]

	f.y = From(tensor.Min(x[0].Data))
	return []*Variable{
		f.y,
	}
}

func (f *MinT) Backward(gy ...*Variable) []*Variable {
	mask := tensor.F2(f.x.Data, f.y.Data, IsClose)
	return []*Variable{
		Mul(gy[0], From(mask)),
	}
}
