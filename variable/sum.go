package variable

import (
	"sort"

	"github.com/itsubaki/autograd/tensor"
)

func Sum(axes ...int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &SumT{
			Axes: axes,
		},
	}).First
}

type SumT struct {
	Axes   []int
	xShape []int
}

func (f *SumT) Forward(x ...*Variable) []*Variable {
	f.xShape = x[0].Shape()

	y := tensor.Sum(x[0].Data, f.Axes...)
	return []*Variable{
		From(y),
	}
}

func (f *SumT) Backward(gy ...*Variable) []*Variable {
	if len(f.Axes) == 0 || len(f.Axes) == len(f.xShape) {
		return []*Variable{
			BroadcastTo(f.xShape...)(gy[0]),
		}
	}

	shape := shapeSum(gy[0].Shape(), f.Axes)
	return []*Variable{
		BroadcastTo(f.xShape...)(Reshape(shape...)(gy[0])),
	}
}

func shapeSum(shape []int, axes []int) []int {
	sort.Ints(axes)
	for _, a := range axes {
		tail := append([]int{1}, shape[a:]...) // insert 1 at axis a
		shape = append(shape[:a], tail...)
	}

	return shape
}
