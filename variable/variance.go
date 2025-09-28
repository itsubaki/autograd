package variable

import "github.com/itsubaki/autograd/tensor"

func Variance(axes ...int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &VarianceT{
			Axes: axes,
		},
	}).First
}

type VarianceT struct {
	Axes []int
	x    *Variable
}

func (f *VarianceT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]

	y := tensor.Variance(x[0].Data, f.Axes...)
	return []*Variable{
		From(y),
	}
}

func (f *VarianceT) Backward(gy ...*Variable) []*Variable {
	if len(f.Axes) == 0 {
		mu := Mean(f.Axes...)(f.x)
		xc := Sub(f.x, mu)

		count := f.x.Data.Size()
		gx := MulC(2/float64(count), xc)
		return []*Variable{
			Mul(gy[0], gx),
		}
	}

	// count
	shape := f.x.Shape()
	count := 1
	for _, ax := range f.Axes {
		count *= shape[ax]
	}

	// mu = mean(x, axes)
	reshape := tensor.KeepDims(shape, f.Axes)
	mu := Mean(f.Axes...)(f.x)
	mu = Reshape(reshape...)(mu)
	xc := Sub(f.x, mu)
	gx := MulC(2/float64(count), xc)

	gy0 := Reshape(reshape...)(gy[0])
	return []*Variable{
		Mul(BroadcastTo(shape...)(gy0), gx),
	}
}
