package variable

import "github.com/itsubaki/autograd/vector"

func AddC(c float64, x ...*Variable) *Variable {
	return (&Function{Forwarder: &AddT{}}).ApplyAndFirst(Const(c), x[0])
}

func Add(x ...*Variable) *Variable {
	return (&Function{Forwarder: &AddT{}}).ApplyAndFirst(x...)
}

type AddT struct{}

func (f *AddT) Forward(x ...*Variable) []*Variable {
	y := vector.Add(x[0].Data, x[1].Data)
	return []*Variable{
		New(y...),
	}
}

func (f *AddT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		gy[0],
		gy[0],
	}
}
