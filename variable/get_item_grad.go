package variable

import (
	"github.com/itsubaki/autograd/tensor"
)

func GetItemGrad(slices, inShape []int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &GetItemGradT{
			Slices:  slices,
			InShape: inShape,
		},
	}).First
}

type GetItemGradT struct {
	Slices  []int
	InShape []int
}

func (f *GetItemGradT) Forward(gy ...*Variable) []*Variable {
	gx := tensor.Zero[float64](f.InShape...)

	// TODO:
	// for i, idx := range f.Slices {
	// 	gx.SetRow(idx, tensor.Add(gx.Row(idx), gy[0].Data.Row(i)))
	// }

	return []*Variable{
		NewFrom(gx),
	}
}

func (f *GetItemGradT) Backward(ggx ...*Variable) []*Variable {
	return []*Variable{
		GetItem(f.Slices)(ggx...),
	}
}
