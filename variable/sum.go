package variable

import "github.com/itsubaki/autograd/vector"

func Sum(x ...*Variable) *Variable {
	return (&Function{Forwarder: &SumT{}}).ApplyAndFirst(x...)
}

type SumT struct {
	x *Variable
}

func (f *SumT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]

	y := vector.Sum(f.x.Data)
	return []*Variable{
		New(y),
	}
}

func (f *SumT) Backward(gy ...*Variable) []*Variable {
	y, _ := vector.Broadcast(gy[0].Data, f.x.Data)
	return []*Variable{
		New(y...),
	}
}
