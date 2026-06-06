package variable

import "github.com/itsubaki/autograd/tensor"

// GetItemGrad returns a function that scatters gradients back to the original shape.
func GetItemGrad(axis int, indices, shape []int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &GetItemGradT{
			Axis:    axis,
			Indices: indices,
			Shape:   shape,
		},
	}).First
}

// GetItemGradT is the differentiable gradient operation for GetItem.
type GetItemGradT struct {
	Axis    int
	Indices []int
	Shape   []int
}

func (f *GetItemGradT) Forward(gy ...*Variable) []*Variable {
	z := tensor.Zeros[float64](f.Shape...)
	gx := tensor.ScatterAdd(z, gy[0].Data, f.Axis, f.Indices)

	return []*Variable{
		From(gx),
	}
}

func (f *GetItemGradT) Backward(ggx ...*Variable) []*Variable {
	return []*Variable{
		GetItem(f.Axis, f.Indices)(ggx...),
	}
}
