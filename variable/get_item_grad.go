package variable

import (
	"github.com/itsubaki/autograd/tensor"
)

func GetItemGrad(indices, inShape []int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &GetItemGradT{
			Indices: indices,
			InShape: inShape,
		},
	}).First
}

type GetItemGradT struct {
	Indices []int
	InShape []int
}

func (f *GetItemGradT) Forward(gy ...*Variable) []*Variable {
	gx := tensor.Zeros[float64](f.InShape...)
	gx.ScatterAdd(gy[0].Data, f.Indices, 0)

	return []*Variable{
		NewFrom(gx),
	}
}

func (f *GetItemGradT) Backward(ggx ...*Variable) []*Variable {
	return []*Variable{
		GetItem(f.Indices)(ggx...),
	}
}
