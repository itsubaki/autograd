package variable

import "github.com/itsubaki/autograd/vector"

func Log(x ...*Variable) *Variable {
	return (&Function{Forwarder: &LogT{}}).ApplyAndFirst(x...)
}

type LogT struct {
	x *Variable
}

func (f *LogT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]

	y := vector.Log(x[0].Data)
	return []*Variable{
		New(y...),
	}
}

func (f *LogT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Div(gy[0], f.x), // gy / x
	}
}
