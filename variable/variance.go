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
		mu := Mean(f.Axes...)(f.x)       // mean(x)
		xc := Sub(f.x, mu)               // x - mean(x)
		count := f.x.Data.Size()         // N
		gx := MulC(2/float64(count), xc) // 2/N * (x - mean(x))
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
	mu := Mean(f.Axes...)(f.x)       // mean(x, axes)
	mu = Reshape(reshape...)(mu)     // reshape
	xc := Sub(f.x, mu)               // x - mean(x, axes)
	gx := MulC(2/float64(count), xc) // 2/N * (x - mean(x, axes))

	gy0 := Reshape(reshape...)(gy[0])
	bgy := BroadcastTo(shape...)(gy0)
	return []*Variable{
		Mul(bgy, gx),
	}
}
