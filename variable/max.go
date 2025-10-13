package variable

import "github.com/itsubaki/autograd/tensor"

func Max(axes ...int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &MaxT{
			Axes: axes,
		},
	}).First
}

type MaxT struct {
	Axes []int
	x, y *Variable
}

func (f *MaxT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]

	f.y = From(tensor.Max(x[0].Data, f.Axes...))
	return []*Variable{
		f.y,
	}
}

func (f *MaxT) Backward(gy ...*Variable) []*Variable {
	if len(f.Axes) == 0 {
		mask := isClose(f.x.Data, f.y.Data)
		return []*Variable{
			Mul(gy[0], From(mask)),
		}
	}

	// shape=[2, 3, 4], axes=[1] -> [2, 1, 4]
	shape := tensor.KeepDims(f.x.Shape(), f.Axes)

	// mask
	y := tensor.Reshape(f.y.Data, shape...)
	mask := isClose(f.x.Data, y)

	// broadcast
	gy0 := Reshape(shape...)(gy[0])
	bgy := BroadcastTo(mask.Shape...)(gy0)
	return []*Variable{
		Mul(bgy, From(mask)),
	}
}

func isClose(a, b *tensor.Tensor[float64]) *tensor.Tensor[float64] {
	return tensor.Float64(tensor.IsClose(a, b))
}
