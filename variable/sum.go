package variable

import "github.com/itsubaki/autograd/vector"

func Sum(x ...*Variable) *Variable {
	return (&Function{Forwarder: &SumT{}}).ApplyAndFirst(x...)
}

type SumT struct {
	Shape int
}

func (f *SumT) Forward(x ...*Variable) []*Variable {
	f.Shape = len(x[0].Data)

	y := vector.Sum(x[0].Data)
	return []*Variable{
		New(y),
	}
}

func (f *SumT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Broadcast(f.Shape)(gy[0]),
	}
}
