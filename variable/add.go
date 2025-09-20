package variable

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/vector"
)

func AddC(c float64, x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &AddT{},
	}).First(New(c), x[0])
}

func Add(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &AddT{},
	}).First(x...)
}

type AddT struct {
	x0Shape []int
	x1Shape []int
}

func (f *AddT) Forward(x ...*Variable) []*Variable {
	f.x0Shape, f.x1Shape = x[0].Shape(), x[1].Shape()

	return []*Variable{
		NewFrom(tensor.Add(x[0].Data, x[1].Data)),
	}
}

func (f *AddT) Backward(gy ...*Variable) []*Variable {
	if vector.Equal(f.x0Shape, f.x1Shape) {
		return []*Variable{
			gy[0],
			gy[0],
		}
	}

	return []*Variable{
		SumTo(f.x0Shape...)(gy[0]),
		SumTo(f.x1Shape...)(gy[0]),
	}
}
