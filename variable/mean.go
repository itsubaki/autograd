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
	shape := f.x.Shape()

	// count
	count := f.x.Data.Size()
	if len(f.Axes) > 0 {
		count = 1
		for _, ax := range f.Axes {
			count *= shape[ax]
		}
	}

	// gx = 1/count * gy
	bgy := BroadcastTo(shape...)(gy...)
	return []*Variable{
		MulC(1/float64(count), bgy),
	}
}
