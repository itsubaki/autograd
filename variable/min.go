package variable

import "github.com/itsubaki/autograd/tensor"

func Min(axes ...int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &MinT{
			Axes: axes,
		},
	}).First
}

type MinT struct {
	Axes []int
	x, y *Variable
}

func (f *MinT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]

	f.y = From(tensor.Min(x[0].Data, f.Axes...))
	return []*Variable{
		f.y,
	}
}

func (f *MinT) Backward(gy ...*Variable) []*Variable {
	if len(f.Axes) == 0 {
		mask := tensor.F2(f.x.Data, f.y.Data, IsClose)
		return []*Variable{
			Mul(gy[0], From(mask)),
		}
	}

	// shape=[2, 3, 4], axes=[1] -> [2, 1, 4]
	shape := tensor.KeepDims(f.x.Shape(), f.Axes)

	// mask
	y := tensor.Reshape(f.y.Data, shape...)
	mask := tensor.F2(f.x.Data, y, IsClose)

	// broadcast
	gy0 := Reshape(shape...)(gy[0])
	bgy := BroadcastTo(mask.Shape...)(gy0)
	return []*Variable{
		Mul(bgy, From(mask)),
	}
}
