package variable

import "github.com/itsubaki/autograd/tensor"

func Sum(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &SumT{},
	}).First(x...)
}

type SumT struct {
	xShape []int
}

func (f *SumT) Forward(x ...*Variable) []*Variable {
	f.xShape = x[0].Shape()

	y := tensor.Sum(x[0].Data)
	return []*Variable{
		From(y),
	}
}

func (f *SumT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		BroadcastTo(f.xShape...)(gy[0]),
	}
}
