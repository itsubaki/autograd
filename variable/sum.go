package variable

import "github.com/itsubaki/autograd/matrix"

func Sum(x ...*Variable) *Variable {
	return (&Function{Forwarder: &SumT{}}).ApplyAndFirst(x...)
}

type SumT struct {
	xShape []int
}

func (f *SumT) Forward(x ...*Variable) []*Variable {
	f.xShape = Shape(x[0])

	y := matrix.Sum(x[0].Data)
	return []*Variable{
		New(y),
	}
}

func (f *SumT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		BroadcastTo(f.xShape...)(gy[0]),
	}
}
