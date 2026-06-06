package variable

import "github.com/itsubaki/autograd/tensor"

// GetItem returns a function that selects items from x[0] along the given axis.
func GetItem(axis int, indices []int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &GetItemT{
			Axis:    axis,
			Indices: indices,
		},
	}).First
}

// GetItemT is the differentiable indexing operation.
type GetItemT struct {
	Axis    int
	Indices []int
	xShape  []int
}

func (f *GetItemT) Forward(x ...*Variable) []*Variable {
	f.xShape = x[0].Shape()

	y := tensor.Take(x[0].Data, f.Axis, f.Indices)
	return []*Variable{
		From(y),
	}
}

func (f *GetItemT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		GetItemGrad(f.Axis, f.Indices, f.xShape)(gy...),
	}
}
