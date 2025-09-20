package variable

import (
	"github.com/itsubaki/autograd/tensor"
)

func SumTo(shape ...int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &SumToT{
			Shape: shape,
		},
	}).First
}

type SumToT struct {
	Shape, xShape []int
}

func (f *SumToT) Forward(x ...*Variable) []*Variable {
	f.xShape = x[0].Shape()

	ax := axes(f.Shape, f.xShape)
	if len(ax) > 0 {
		y := tensor.Sum(x[0].Data, ax...)
		return []*Variable{
			NewFrom(tensor.Reshape(y, f.Shape...)),
		}
	}

	return []*Variable{
		NewFrom(tensor.Reshape(x[0].Data, f.Shape...)),
	}
}

func (f *SumToT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		BroadcastTo(f.xShape...)(gy[0]),
	}
}

func axes(a, b []int) []int {
	if len(a) < len(b) {
		diff := len(b) - len(a)
		newA := make([]int, len(b))
		for i := range diff {
			newA[i] = 1
		}

		copy(newA[diff:], a)
		a = newA
	}

	var axes []int
	for i := range a {
		if a[i] == 1 && b[i] > 1 {
			axes = append(axes, i)
		}
	}

	return axes
}
