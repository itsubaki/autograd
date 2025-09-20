package variable

import "github.com/itsubaki/autograd/tensor"

func GetItem(indices []int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &GetItemT{
			Indices: indices,
		},
	}).First
}

type GetItemT struct {
	Indices []int
	xShape  []int
}

func (f *GetItemT) Forward(x ...*Variable) []*Variable {
	f.xShape = x[0].Shape()

	return []*Variable{
		NewFrom(tensor.Take(x[0].Data, f.Indices, 0)),
	}
}

func (f *GetItemT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		GetItemGrad(f.Indices, f.xShape)(gy...),
	}
}
