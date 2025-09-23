package variable

import "github.com/itsubaki/autograd/tensor"

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

	// shape=[2, 3, 4], axes=[1] -> [2, 1, 4]
	shape := keepDims(f.xShape, f.Axes)

	// broadcast
	gy0 := Reshape(shape...)(gy[0])
	bgy := BroadcastTo(f.xShape...)(gy0)
	return []*Variable{
		bgy,
	}
}
