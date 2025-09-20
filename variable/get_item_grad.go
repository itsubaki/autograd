package variable

import "github.com/itsubaki/autograd/tensor"

func GetItemGrad(indices, shape []int, axis int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &GetItemGradT{
			Indices: indices,
			Shape:   shape,
			Axis:    axis,
		},
	}).First
}

type GetItemGradT struct {
	Indices []int
	Shape   []int
	Axis    int
}

func (f *GetItemGradT) Forward(gy ...*Variable) []*Variable {
	gx := tensor.Zeros[float64](f.Shape...)
	gx.ScatterAdd(gy[0].Data, f.Indices, f.Axis)

	return []*Variable{
		NewFrom(gx),
	}
}

func (f *GetItemGradT) Backward(ggx ...*Variable) []*Variable {
	return []*Variable{
		GetItem(f.Indices, f.Axis)(ggx...),
	}
}
