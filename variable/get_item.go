package variable

import "github.com/itsubaki/autograd/tensor"

// GetItem returns a function that selects items from x[0] along the given axis.
func GetItem(indices []int, axis int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &GetItemT{
			Indices: indices,
			Axis:    axis,
		},
	}).First
}

// GetItemT is the differentiable indexing operation.
type GetItemT struct {
	Indices []int
	Axis    int
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
		GetItemGrad(f.Indices, f.xShape, f.Axis)(gy...),
	}
}
