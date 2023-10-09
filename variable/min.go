package variable

import "github.com/itsubaki/autograd/matrix"

func Min(x ...*Variable) *Variable {
	return (&Function{Forwarder: &MinT{}}).ApplyAndFirst(x...)
}

type MinT struct {
	MaxT
}

func (f *MinT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]

	f.y = NewOf(matrix.Min(x[0].Data)...)
	return []*Variable{
		f.y,
	}
}