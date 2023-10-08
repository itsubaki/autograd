package variable

import (
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/vector"
)

func GetItemGrad(slices, inShape []int) func(x ...*Variable) *Variable {
	return (&Function{Forwarder: &GetItemGradT{Slices: slices, InShape: inShape}}).ApplyAndFirst
}

type GetItemGradT struct {
	Slices  []int
	InShape []int
}

func (f *GetItemGradT) Forward(gy ...*Variable) []*Variable {
	gx := matrix.Zero(f.InShape[0], f.InShape[1])
	for i, idx := range f.Slices {
		gx[idx] = vector.Add(gx[idx], gy[0].Data[i])
	}

	return []*Variable{
		NewOf(gx...),
	}
}

func (f *GetItemGradT) Backward(ggx ...*Variable) []*Variable {
	return []*Variable{
		GetItem(f.Slices)(ggx...),
	}
}
