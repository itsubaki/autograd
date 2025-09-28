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
	// NOTE: can't double backprop
	sp := tensor.Split(gy[0].Data, f.size, f.Axis)

	gx := make([]*Variable, len(sp))
	for i, v := range sp {
		gx[i] = From(v)
	}

	return gx
}
