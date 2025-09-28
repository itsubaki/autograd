package variable

import "github.com/itsubaki/autograd/tensor"

func Concat(axis int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &ConcatT{
			Axis: axis,
		},
	}).First
}

type ConcatT struct {
	Axis int
	size []int
}

func (f *ConcatT) Forward(x ...*Variable) []*Variable {
	list := make([]*tensor.Tensor[float64], 0)
	for _, v := range x {
		list = append(list, v.Data)
		f.size = append(f.size, v.Shape()[f.Axis])
	}

	y := tensor.Concat(list, f.Axis)
	return []*Variable{
		From(y),
	}
}

func (f *ConcatT) Backward(gy ...*Variable) []*Variable {
	return Split(f.size, f.Axis)(gy[0])
}
