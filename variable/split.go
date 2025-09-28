package variable

import "github.com/itsubaki/autograd/tensor"

func Split(size []int, axis int) func(x ...*Variable) []*Variable {
	return (&Function{
		Forwarder: &SplitT{
			Size: size,
			Axis: axis,
		},
	}).Forward
}

type SplitT struct {
	Size []int
	Axis int
	y    []*Variable
}

func (f *SplitT) Forward(x ...*Variable) []*Variable {
	// split
	y := tensor.Split(x[0].Data, f.Size, f.Axis)

	f.y = make([]*Variable, len(y))
	for i, v := range y {
		f.y[i] = From(v)
	}

	return f.y
}

func (f *SplitT) Backward(gy ...*Variable) []*Variable {
	list := make([]*Variable, len(f.Size))
	for i := range f.Size {
		if gy[i] == nil {
			list[i] = ZeroLike(f.y[i])
			continue
		}

		list[i] = gy[i]
	}

	return []*Variable{
		Concat(f.Axis)(list...),
	}
}
