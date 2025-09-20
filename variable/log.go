package variable

import "github.com/itsubaki/autograd/tensor"

func Log(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &LogT{},
	}).First(x...)
}

type LogT struct {
	x *Variable
}

func (f *LogT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]

	return []*Variable{
		NewFrom(tensor.Log(x[0].Data)),
	}
}

func (f *LogT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Div(gy[0], f.x), // gy / x
	}
}
