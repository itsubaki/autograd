package variable

import "github.com/itsubaki/autograd/tensor"

func Mean(axes ...int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &MeanT{
			Axes: axes,
		},
	}).First
}

type MeanT struct {
	Axes []int
	x    *Variable
}

func (f *MeanT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]

	return []*Variable{
		From(tensor.Mean(x[0].Data, f.Axes...)),
	}
}

func (f *MeanT) Backward(gy ...*Variable) []*Variable {
	if len(f.Axes) == 0 {
		size := f.x.Data.Size()
		bgy := BroadcastTo(f.x.Shape()...)(gy[0])
		return []*Variable{
			MulC(1/float64(size), bgy),
		}
	}

	// size
	shape := f.x.Shape()
	size := 1
	for _, ax := range f.Axes {
		size *= shape[ax]
	}

	// gx = 1/N * gy
	reshape := tensor.KeepDims(shape, f.Axes)
	gy0 := Reshape(reshape...)(gy[0])
	bgy := BroadcastTo(shape...)(gy0)
	return []*Variable{
		MulC(1/float64(size), bgy),
	}
}
